package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.stream.DoubleStream;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link GlobMeanMax}
 *
 * @author Christian Skärby
 */
public class GlobMeanMaxTest {

    /**
     * Test that global mean max is correctly calculated
     */
    @Test
    public void addLayersGraph() {

        final double[] inputArr = new double[]{0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0};
        final INDArray input = Nd4j.create(1, 1, inputArr.length, 1);
        for (int i = 0; i < input.length(); i++) {
            input.putScalar(new int[]{0, 0, i, 0}, inputArr[i]);
        }
        final LayerBlockConfig toTest = new GlobMeanMax();

        final ComputationGraph graph = MockGraphAdapter.createRealComputationGraph(
                toTest,
                1,
                InputType.convolutional(input.size(2), input.size(3), input.size(1)));

        final double mean = DoubleStream.of(inputArr).summaryStatistics().getAverage();
        final double max = DoubleStream.of(inputArr).max().getAsDouble();
        assertEquals("Incorrect output!", (max + mean) / 2, graph.output(input)[0].getDouble(0), 1e-5); // Half precision or what?
    }
}