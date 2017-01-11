#ifndef CAFFE_RPN_LAYER_HPP_
#define CAFFE_RPN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include"opencv2/opencv.hpp"
#define mymax(a,b) ((a)>(b))?(a):(b)
#define mymin(a,b) ((a)>(b))?(b):(a)
namespace caffe {

	/**
	* @brief implement RPN layer for faster rcnn
	*/

	template <typename Dtype>
	class RPNLayer : public Layer<Dtype> {
	public:
		explicit RPNLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top){}
		virtual inline const char* type() const { return "RPN"; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		//virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			//const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};

		private:
		
		int feat_stride_;
		int base_size_;
		vector<int> anchor_scales_;
		vector<float> ratios_;
		vector<vector<float>> gen_anchors_;
		int anchors_[9][4];
		int anchors_nums_;
		int min_size_;
		int pre_nms_topN_;
		int post_nms_topN_;
		float nms_thresh_;
		int src_height_;
		int src_width_;
		float src_scale_;
		private:
		void generate_anchors();
		vector<vector<float>> ratio_enum(vector<float>);
		vector<float> whctrs(vector<float>);
		vector<float> mkanchor(float w,float h,float x_ctr,float y_ctr);
		vector<vector<float>> scale_enum(vector<float>);
		cv::Mat proposal_local_anchor(int width, int height);
		
		void filter_boxs(cv::Mat& pre_box, cv::Mat& score, vector<RPN::abox>& aboxes);
		
	};
}  // namespace caffe

#endif  // CAFFE_RPN_LAYER_HPP_