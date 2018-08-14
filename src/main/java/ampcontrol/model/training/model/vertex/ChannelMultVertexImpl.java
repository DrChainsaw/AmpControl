package ampcontrol.model.training.model.vertex;


import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

/**
 * {@link BaseGraphVertex} which multiplies each channel in a convolutional activation of size [b,c,h,w] with a scalar
 * of size (b,c) where b is the batch size, c is the number of channels, h is the height and w is the width. Used in
 * squeeze-exitation networks: https://arxiv.org/abs/1709.01507
 *
 * @author Christian Skärby
 */
public class ChannelMultVertexImpl extends BaseGraphVertex {

    public ChannelMultVertexImpl(ComputationGraph graph, String name, int vertexIndex) {
        this(graph, name, vertexIndex, null, null);

    }

    public ChannelMultVertexImpl(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices,
                                 VertexIndices[] outputVertices) {
        super(graph, name, vertexIndex, inputVertices, outputVertices);
    }

    @Override
    public boolean hasLayer() {
        return false;
    }

    @Override
    public boolean isOutputVertex() {
        return false;
    }

    @Override
    public Layer getLayer() {
        return null;
    }

    @Override
    public INDArray doForward(boolean training, LayerWorkspaceMgr workspaceMgr) {
        if (!canDoForward())
            throw new IllegalStateException("Cannot do forward pass: inputs not set " + this.toString());

        if (inputs.length != 2)
            throw new IllegalArgumentException(
                    this.toString() + " only supports 2 inputs.");

        INDArray prod = workspaceMgr.dup(ArrayType.ACTIVATIONS, inputs[0]);
        INDArray perCh = inputs[1];

        if (perCh.rank() != 2) {
            throw new IllegalArgumentException(
                    this.toString() + " must have rank 2 inputs! " + perCh.rank());
        }

        //  double timeOuter1;
        //   double timeOuter2;

        // timeOuter1 = System.nanoTime();
        INDArray result = scaleChannels(prod, perCh);
        //  timeOuter2 = System.nanoTime();
        //  System.out.println("channel receive: " + (timeOuter2 - timeOuter1) / 1e6);

        return result;
    }

    private INDArray scaleChannels(INDArray channelActivations, INDArray scaleFactorsPerChannel) {
        // channelActivations with shape [b, c, h, w] with
        // b: Batch size
        // c: Number of channels
        // h: Height of input
        // w: Width of input

        //scaleFactorsPerChannel with shape [b, c] with
        // b: Batch size
        // c: Number of channels

        // Goal: multiply each h*w activation a_i with the corresponding scale factor s_i, i in range [0, b*c-1]

        final long nrofChannelBatch = channelActivations.tensorssAlongDimension(2, 3); // view each channel and each batch
        final long nrofFeaturesInActivation = channelActivations.tensorssAlongDimension(0, 1); // view each h and w activation for all batches

        // From empiric testing: Whatever makes the fewest number of loops is fastest
        if (nrofChannelBatch < nrofFeaturesInActivation) {
            //  double time1 = System.nanoTime();
            // Alt1: Loop over each channel and batch and multiply the resulting h*w with each scalar s
            for (int i = 0; i < nrofChannelBatch; i++) {
                INDArray tens = channelActivations.tensorAlongDimension(i, 2, 3);
                tens.muli(scaleFactorsPerChannel.getDouble(i));
            }
            // double  time2 = System.nanoTime();
            // System.out.println("loop scal mult: " + (time2 - time1) / 1e6);
        } else {
            // double time1 = System.nanoTime();
            // Alt2: Loop over all h*w individual elements in the activation (shape [b,c]) and do elem wise multiplication
            // Multiply each individual channel activation for all batches and channels
            for (int i = 0; i < nrofFeaturesInActivation; i++) {
                INDArray tens = channelActivations.tensorAlongDimension(i, 0, 1);
                tens.muli(scaleFactorsPerChannel);
            }
            //  double time2 = System.nanoTime();
            // System.out.println("loop vec mult: " + (time2 - time1) / 1e6);
        }


        return channelActivations;
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        if (!canDoBackward())
            throw new IllegalStateException("Cannot do backward pass: errors not set " + this.toString());

        // F = function this block performs (scalar multiplication of channel activations)
        // u = channel activations (inputs[0])
        // s = scalars (inputs[1])
        // epsilon = dL / dF
        // F = s * u
        // dL / du = dL / dF * dF/du
        // dL / ds = dL / dF * dF/ds
        // dF/du = s
        // dF/s = u
        // dL / du = epsilon * s // Same op as forward
        // dL / ds = epsilon * u // How to make this a scalar value??

        INDArray eps_scale = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilon).muli(inputs[0]);
        // Error for each scale factor is sum of errors. Can't find a rational reason why though except other ways
        // either don't work dimensionally or just take me to NaNaNaNaN-land.
        // From my attempts to decipher paper authors caffe code this is what they are doing too
        // https://github.com/hujie-frank/SENet/blob/master/src/caffe/layers/axpy_layer.cpp
        INDArray out_scale = eps_scale.sum(2, 3);
        out_scale = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, out_scale);

        // Take a step of size mean(abs(eps_scale)) in the "dominant" direction of the error -> NaNaNaNaN-land
        //INDArray sum = eps_scale.sum(2,3);
        //INDArray sign = sum.div(abs(sum));
        //INDArray backSe = eps_scale.amean(2, 3).muli(sign);

        INDArray out_conv = scaleChannels(workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilon), inputs[1]);
        return new Pair<>(null, new INDArray[]{out_conv, out_scale});
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        if (backpropGradientsViewArray != null)
            throw new IllegalArgumentException(
                    "Vertex does not have gradients; gradients view array cannot be set here " + this.toString());
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName() + "(id=" + this.getVertexIndex() + ",name=\"" + this.getVertexName() + "\")";
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState,
                                                           int minibatchSize) {
        //No op
        if (maskArrays == null || maskArrays.length == 0) {
            return null;
        }

        return new Pair<>(maskArrays[0], currentMaskState);
    }
}
