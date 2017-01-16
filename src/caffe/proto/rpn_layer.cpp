#include <algorithm>
#include <vector>

#include "caffe/layers/rpn_layer.hpp"
#include "caffe/util/math_functions.hpp"

int debug = 0;
int  tmp[9][4] = {
	{ -83, -39, 100, 56 },
	{ -175, -87, 192, 104 },
	{ -359, -183, 376, 200 },
	{ -55, -55, 72, 72 },
	{ -119, -119, 136, 136 },
	{ -247, -247, 264, 264 },
	{ -35, -79, 52, 96 },
	{ -79, -167, 96, 184 },
	{ -167, -343, 184, 360 }
};
namespace caffe {

	template <typename Dtype>
	void RPNLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		feat_stride_ = layer_param_.rpn_param().feat_stride();
		base_size_ = layer_param_.rpn_param().basesize();
		min_size_ = layer_param_.rpn_param().boxminsize();
		pre_nms_topN_ = layer_param_.rpn_param().per_nms_topn();
		post_nms_topN_ = layer_param_.rpn_param().post_nms_topn();
		nms_thresh_ = layer_param_.rpn_param().nms_thresh();
		int scales_num = layer_param_.rpn_param().scale_size();
		for (int i = 0; i < scales_num; ++i)
		{
			anchor_scales_.push_back(layer_param_.rpn_param().scale(i));
		}
		int ratios_num = layer_param_.rpn_param().ratio_size();
		for (int i = 0; i < ratios_num; ++i)
		{
			ratios_.push_back(layer_param_.rpn_param().ratio(i));
		}
		generate_anchors();
		
		//memcpy(anchors_, tmp, 9 * 4 * sizeof(int));
		//anchors_nums_ = 9;
		for (int i = 0; i<gen_anchors_.size(); ++i)
		{
			for (int j = 0; j<gen_anchors_[i].size(); ++j)
			{
				anchors_[i][j] = gen_anchors_[i][j];
				if (debug)
					LOG(INFO) << anchors_[i][j] << std::endl;;
			}
		}
		anchors_nums_ = gen_anchors_.size();
		
		top[0]->Reshape(1, 5, 1, 1);
		if (top.size() > 1)
		{
			top[1]->Reshape(1, 1, 1, 1);
		}
	}

	template <typename Dtype>
	void RPNLayer<Dtype>::generate_anchors(){
		//generate base anchor
		vector<float> base_anchor;
		base_anchor.push_back(0);
		base_anchor.push_back(0);
		base_anchor.push_back(base_size_ - 1);
		base_anchor.push_back(base_size_ - 1);
		//enum ratio anchors
		vector<vector<float>>ratio_anchors = ratio_enum(base_anchor);
		for (int i = 0; i < ratio_anchors.size(); ++i)
		{
			vector<vector<float>> tmp = scale_enum(ratio_anchors[i]);
			gen_anchors_.insert(gen_anchors_.end(), tmp.begin(), tmp.end());
		}
	}

	template <typename Dtype>
	vector<vector<float>> RPNLayer<Dtype>::scale_enum(vector<float> anchor){
		vector<vector<float>> result;
		vector<float> reform_anchor = whctrs(anchor);
		float x_ctr = reform_anchor[2];
		float y_ctr = reform_anchor[3];
		float w = reform_anchor[0];
		float h = reform_anchor[1];
		for (int i = 0; i < anchor_scales_.size(); ++i)
		{
			float ws = w * anchor_scales_[i];
			float hs = h *  anchor_scales_[i];
			vector<float> tmp = mkanchor(ws, hs, x_ctr, y_ctr);
			result.push_back(tmp);
		}
		return result;
	}


	template <typename Dtype>
	vector<vector<float> > RPNLayer<Dtype>::ratio_enum(vector<float> anchor){
		vector<vector<float>> result;
		vector<float> reform_anchor = whctrs(anchor);
		float x_ctr = reform_anchor[2];
		float y_ctr = reform_anchor[3];
		float size = reform_anchor[0] * reform_anchor[1];
		for (int i = 0; i < ratios_.size(); ++i)
		{
			float size_ratios = size / ratios_[i];
			float ws = round(sqrt(size_ratios));
			float hs = round(ws*ratios_[i]);
			vector<float> tmp = mkanchor(ws, hs, x_ctr, y_ctr);
			result.push_back(tmp);
		}
		return result;
	}

	template <typename Dtype>
	vector<float> RPNLayer<Dtype>::mkanchor(float w, float h, float x_ctr, float y_ctr){
		vector<float> tmp;
		tmp.push_back(x_ctr - 0.5*(w - 1));
		tmp.push_back(y_ctr - 0.5*(h - 1));
		tmp.push_back(x_ctr + 0.5*(w - 1));
		tmp.push_back(y_ctr + 0.5*(h - 1));
		return tmp;
	}
	template <typename Dtype>
	vector<float> RPNLayer<Dtype>::whctrs(vector<float> anchor){
		vector<float> result;
		result.push_back(anchor[2] - anchor[0] + 1); //w
		result.push_back(anchor[3] - anchor[1] + 1); //h
		result.push_back((anchor[2] + anchor[0]) / 2); //ctrx
		result.push_back((anchor[3] + anchor[1]) / 2); //ctry
		return result;
	}
	

	template <typename Dtype>
	cv::Mat RPNLayer<Dtype>::proposal_local_anchor(int width, int height)
	{
		Blob<float> shift;
		cv::Mat shitf_x(height, width, CV_32SC1);
		cv::Mat shitf_y(height, width, CV_32SC1);
		for (size_t i = 0; i < width; i++)
		{
			for (size_t j = 0; j < height; j++)
			{
				shitf_x.at<int>(j, i) = i * feat_stride_;
				shitf_y.at<int>(j, i) = j * feat_stride_;
			}
		}
		shift.Reshape(anchors_nums_, width*height, 4,  1);
		float *p = shift.mutable_cpu_diff(), *a = shift.mutable_cpu_data();
		for (int i = 0; i < height*width; i++)
		{
			for (int j = 0; j < anchors_nums_; j++)
			{
				size_t num = i * 4 + j * 4 * height*width;
				p[num + 0] = -shitf_x.at<int>(i / shitf_x.cols, i % shitf_x.cols);
				p[num + 2] = -shitf_x.at<int>(i / shitf_x.cols, i % shitf_x.cols);
				p[num + 1] = -shitf_y.at<int>(i / shitf_y.cols, i % shitf_y.cols);
				p[num + 3] = -shitf_y.at<int>(i / shitf_y.cols, i % shitf_y.cols);
				a[num + 0] = anchors_[j][0];
				a[num + 1] = anchors_[j][1];
				a[num + 2] = anchors_[j][2];
				a[num + 3] = anchors_[j][3];
			}
		}
		shift.Update();
		cv::Mat loacl_anchors(anchors_nums_ * height*width, 4, CV_32FC1);
		size_t num = 0;
		for (int i = 0; i < height; ++i)
		{
			for (int j = 0; j < width; ++j)
			{
				for (int c = 0; c < anchors_nums_; ++c)
				{
					for (int k = 0; k < 4; ++k)
					{
						loacl_anchors.at<float>((i*width + j)*anchors_nums_+c, k)= shift.data_at(c, i*width + j, k, 0);
					}
				}
			}
		}
		return loacl_anchors;
	}

	

	template<typename Dtype>
	void RPNLayer<Dtype>::filter_boxs(cv::Mat& pre_box, cv::Mat& score, vector<RPN::abox>& aboxes)
	{
		float localMinSize=min_size_*src_scale_;
		aboxes.clear();
		
		for (int i = 0; i < pre_box.rows; i++)
		{
			int widths = pre_box.at<float>(i, 2) - pre_box.at<float>(i, 0) + 1;
			int heights = pre_box.at<float>(i, 3) - pre_box.at<float>(i, 1) + 1;
			if (widths >= localMinSize || heights >= localMinSize)
			{
				RPN::abox tmp;
				tmp.x1 = pre_box.at<float>(i, 0);
				tmp.y1 = pre_box.at<float>(i, 1);
				tmp.x2 = pre_box.at<float>(i, 2);
				tmp.y2 = pre_box.at<float>(i, 3);
				tmp.score = score.at<float>(i, 0);
				aboxes.push_back(tmp);
			}
		}
	}

	
	
	/*template <typename Dtype>
	void RPNLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		this->Forward_cpu(bottom, top);
	}*/

	template <typename Dtype>
	void RPNLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		int width = bottom[1]->width();
		int height = bottom[1]->height();
		//int channels = bottom[1]->channels();


		//get boxs_delta,向右。
		cv::Mat boxs_delta(height*width*anchors_nums_, 4, CV_32FC1);
		for (int i = 0; i < height; ++i)
		{
			for (int j = 0; j < width; ++j)
			{
				for (int k = 0; k < anchors_nums_; ++k)
				{
					for (int c = 0; c < 4; ++c)
					{
						boxs_delta.at<float>((i*width + j)*anchors_nums_ + k, c) = bottom[1]->data_at(0, k*4 + c, i, j);
					}
				}
			}
		}

		//get sores 向右，前面anchors_nums_个位bg的得分，后面anchors_nums_为fg得分，我们需要的是后面的。
		cv::Mat scores(height*width*anchors_nums_, 1, CV_32FC1);
		for (int i = 0; i < height; ++i)
		{
			for (int j = 0; j < width; ++j)
			{
				for (int k = 0; k < anchors_nums_; ++k)
				{
					scores.at<float>((i*width + j)*anchors_nums_+k, 0) = bottom[0]->data_at(0, k + anchors_nums_, i, j);
				}
			}
		}

		//get im_info

		src_height_ = bottom[2]->data_at(0, 0,0,0);
		src_width_ = bottom[2]->data_at(0, 1,0,0);
		src_scale_ = bottom[2]->data_at(0, 2, 0, 0);

		//gen local anchors 向右
		cv::Mat local_anchors = proposal_local_anchor(width, height);

		//Convert anchors into proposals via bbox transformations
		cv::Mat pre_box = RPN::bbox_tranform_inv(local_anchors, boxs_delta);

		for (int i = 0; i < pre_box.rows; ++i)
		{
			if (pre_box.at<float>(i, 0) < 0)	pre_box.at<float>(i, 0) = 0;
			if (pre_box.at<float>(i, 0) > (src_width_ - 1))	pre_box.at<float>(i, 0) = src_width_ - 1;
			if (pre_box.at<float>(i, 2) < 0)	pre_box.at<float>(i, 2) = 0;
			if (pre_box.at<float>(i, 2) > (src_width_ - 1))	pre_box.at<float>(i, 2) = src_width_ - 1;

			if (pre_box.at<float>(i, 1) < 0)	pre_box.at<float>(i, 1) = 0;
			if (pre_box.at<float>(i, 1) > (src_height_ - 1))	pre_box.at<float>(i, 1) = src_height_ - 1;
			if (pre_box.at<float>(i, 3) < 0)	pre_box.at<float>(i, 3) = 0;
			if (pre_box.at<float>(i, 3) > (src_height_ - 1))	pre_box.at<float>(i, 3) = src_height_ - 1;
		}
		vector<RPN::abox>aboxes;
		filter_boxs(pre_box, scores, aboxes);
		std::sort(aboxes.rbegin(), aboxes.rend()); //降序
		if (pre_nms_topN_ > 0)
		{
			int tmp = mymin(pre_nms_topN_, aboxes.size());
			aboxes.erase(aboxes.begin() + tmp - 1, aboxes.end());
		}
		RPN::nms(aboxes,nms_thresh_);

		if (post_nms_topN_ > 0)
		{
			int tmp = mymin(post_nms_topN_, aboxes.size());
			aboxes.erase(aboxes.begin() + tmp - 1, aboxes.end());
		}
		top[0]->Reshape(aboxes.size(),5,1,1);
		Dtype *top0 = top[0]->mutable_cpu_data();
		for (int i = 0; i < aboxes.size(); ++i)
		{
			top0[0] = 0;
			top0[1] = aboxes[i].x1;
			top0[2] = aboxes[i].y1; 
			top0[3] = aboxes[i].x2;
			top0[4] = aboxes[i].y2;
			top0 += top[0]->offset(1);
		}
		if (top.size()>1)
		{
			top[1]->Reshape(aboxes.size(), 1,1,1);
			Dtype *top1 = top[1]->mutable_cpu_data();
			for (int i = 0; i < aboxes.size(); ++i)
			{
				top1[0] = aboxes[i].score;
				top1 += top[1]->offset(1);
			}
		}
	}

#ifdef CPU_ONLY
		STUB_GPU(RPNLayer);
#endif

	INSTANTIATE_CLASS(RPNLayer);
	REGISTER_LAYER_CLASS(RPN);

}  // namespace caffe