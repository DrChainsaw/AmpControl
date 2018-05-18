package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link MinMaxPool}
 *
 * @author Christian Skärby
 */
public class MinMaxPoolTest {

    /**
     * Test that output is correct
     */
    @Test
    public void addLayers() {
        final double[] inputArr = {-1, 1, 2, 3};
        final double[] expected = {2, 3, -1, 1}; // max(inputArr[0:2]), max(inputArr[1:3]), min(inputArr[0:2]), min(inputArr[1:3])
        final INDArray input = Nd4j.create(1, 1, inputArr.length, 1);
        for (int i = 0; i < input.length(); i++) {
            input.putScalar(new int[]{0, 0, i, 0}, inputArr[i]);
        }


        final LayerBlockConfig toTest = new MinMaxPool()
                .setSize_h(3)
                .setSize_w(1)
                .setStride(1);
        final ComputationGraph graph = MockGraphAdapter.createRealComputationGraph(toTest,
                2, // Bit of a hack to make it four outputs since pooling layer does not figure out number of outputs for dense
                InputType.convolutional(input.size(2), input.size(3), input.size(1)));

        DummyOutputLayer.setEyeOutput(graph);

        assertEquals("Incorrect output", Nd4j.create(expected), graph.output(input)[0]);
    }
}