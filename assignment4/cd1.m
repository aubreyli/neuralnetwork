function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.

% rbm_w:			100 X 256
% visible_state:	256 X 37

	visible_data = sample_bernoulli(visible_data);
	hidden_prob = sample_bernoulli(visible_state_to_hidden_probabilities(rbm_w, visible_data));
	
	visible_data1 = sample_bernoulli(hidden_state_to_visible_probabilities(rbm_w, hidden_prob));
	%hidden_prob1 = sample_bernoulli(visible_state_to_hidden_probabilities(rbm_w, visible_data1));
    hidden_prob1 = visible_state_to_hidden_probabilities(rbm_w, visible_data1);
	
	d_gradient = configuration_goodness_gradient(visible_data, hidden_prob);
	d_gradient1 = configuration_goodness_gradient(visible_data1, hidden_prob1);
    ret = d_gradient - d_gradient1;
	
end
