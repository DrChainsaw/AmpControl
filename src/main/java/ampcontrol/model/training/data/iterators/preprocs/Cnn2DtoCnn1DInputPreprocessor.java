package ampcontrol.model.training.data.iterators.preprocs;

import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

/**
 * {@link InputPreProcessor} which changes CNN 2D input to CNN 1D input. Assumes only one channel.
 *
 * @author Christian Skärby
 */
public class Cnn2DtoCnn1DInputPreprocessor implements InputPreProcessor {

    /**
     *
     */
    private static final long serialVersionUID = 5008669952802464315L;


    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        //featureArr = Nd4j.create(new int[] {batchSize, 1, feature.length, feature[0].length}, 'f');
        //For input shape [a,b,c,d], tensorssAlongDimension(0,2,3) gives b tensors, and tensorAlongDimension(i,0,2,3) returns tensors of shape [a,c,d].
        //  INDArray output = input.tensorAlongDimension(0,0, 3, 2);
        // int[] outputShape = output.shape();
        //  int[] inputShape = input.shape();
        if(input.size(1) > 1) {
            // Implementation could probably be reworked to do something like [miniBatchSize, width*channels, height]
            throw new UnsupportedOperationException("Hacky implementation assumes single channel only. Got " + input.size(1));
        }
        INDArray rnnFeatures = input.tensorAlongDimension(0, 0, 3, 2);
        //int[] shape = rnnFeatures.shape();
        return rnnFeatures;

    }

    @Override
    public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        return output;
    }

    @Override
    public InputPreProcessor clone() {
        return null;
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        //  if(inputType instanceof InputType.InputTypeConvolutional) {
        //  InputType.InputTypeConvolutional conv
        //  }
        //  return InputType.InputTypeRecurrent.recurrent()
        return inputType;
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        return null;
    }

}