package ampcontrol.model.training.model.vertex;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link ChannelMultVertex}
 *
 * @author Christian Skärby
 */
public class ChannelMultVertexTest {

    /**
     * Test correct inputs
     */
    @Test
    public void setInputsCorrect() {
        final int nrofChannels = 37;
        final InputType input1 = InputType.convolutional(17, 9, nrofChannels);
        final InputType input2 = InputType.feedForward(nrofChannels);
        final InputType output = new ChannelMultVertex().getOutputType(777, input1, input2);
        assertEquals("Incorrect output type!", input1, output);
    }

    /**
     * Test inputs switched order
     */
    @Test(expected = InvalidInputTypeException.class)
    public void setInputsSwitchedOrder() {
        final int nrofChannels = 37;
        final InputType input1 = InputType.convolutional(17, 9, nrofChannels);
        final InputType input2 = InputType.feedForward(nrofChannels);
        new ChannelMultVertex().getOutputType(777, input2, input1);
    }

    /**
     * Test inputs misaligned size
     */
    @Test(expected = InvalidInputTypeException.class)
    public void setInputsMisalignedSize() {
        final int nrofChannels = 37;
        final InputType input1 = InputType.convolutional(17, 9, nrofChannels);
        final InputType input2 = InputType.feedForward(nrofChannels+1);
        new ChannelMultVertex().getOutputType(777, input2, input1);
    }

}