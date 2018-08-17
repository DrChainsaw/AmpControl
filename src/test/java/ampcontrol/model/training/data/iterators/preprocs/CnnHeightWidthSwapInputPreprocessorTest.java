package ampcontrol.model.training.data.iterators.preprocs;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link CnnHeightWidthSwapInputPreprocessor}.
 *
 * @author Christian Skärby
 */
public class CnnHeightWidthSwapInputPreprocessorTest {

    private static final LayerWorkspaceMgr wsMgr = new LayerWorkspaceMgr.Builder().defaultNoWorkspace().build();

    /**
     * Test preProcess
     */
    @Test
    public void preProcess() {
        final int[] shapeIn = {3, 5, 7, 11};
        final long[] expectedShape = {3, 5, 11, 7};
        final InputPreProcessor pp = new CnnHeightWidthSwapInputPreprocessor();
        assertArrayEquals("Incorrect shape!",expectedShape, pp.preProcess(Nd4j.create(shapeIn), shapeIn[0], wsMgr).shape());
    }

    /**
     * Test backProp
     */
    @Test
    public void backprop() {
        final int[] shapeIn = {3, 5, 7, 11};
        final long[] expectedShape = {3, 5, 11, 7};
        final InputPreProcessor pp = new CnnHeightWidthSwapInputPreprocessor();
        assertArrayEquals("Incorrect shape!",expectedShape, pp.preProcess(Nd4j.create(shapeIn), shapeIn[0], wsMgr).shape());
    }

    /**
     * Test getOutputType
     */
    @Test
    public void getOutputType() {
        final InputType typeIn = InputType.convolutional(3,5,7);
        final InputType expectedType = InputType.convolutional(5,3,7);
        final InputPreProcessor pp = new CnnHeightWidthSwapInputPreprocessor();
        assertEquals("Incorrect inputType!", expectedType, pp.getOutputType(typeIn));
    }
}