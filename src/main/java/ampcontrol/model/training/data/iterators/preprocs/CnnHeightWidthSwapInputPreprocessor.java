package ampcontrol.model.training.data.iterators.preprocs;

import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import javax.ws.rs.NotSupportedException;

/**
 * {@link InputPreProcessor} which swaps height and width dimensions of CNN input. Intended use is when doing 1D
 * convolutions with 2D convolutional layers.
 *
 * @author Christian Skärby
 */
public class CnnHeightWidthSwapInputPreprocessor implements InputPreProcessor {
    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        //int[] shapePre = input.shape();
        INDArray out = input.swapAxes(2,3);
        //int[] shapePost = out.shape();
        return out;
    }

    @Override
    public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        return output.swapAxes(2,3);
    }

    @Override
    public InputPreProcessor clone() {
        return new CnnHeightWidthSwapInputPreprocessor();
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        if(inputType.getType() == InputType.Type.CNN) {
            InputType.InputTypeConvolutional cnnInput = (InputType.InputTypeConvolutional)inputType;
            return new InputType.InputTypeConvolutional(cnnInput.getWidth(), cnnInput.getHeight(), cnnInput.getChannels());
        }
        throw new IllegalArgumentException("Only works for CNN types! Was " + inputType);
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        throw new NotSupportedException("Not implemented yet!");
    }
}