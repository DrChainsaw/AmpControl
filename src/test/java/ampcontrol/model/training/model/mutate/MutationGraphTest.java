package ampcontrol.model.training.model.mutate;

import com.google.common.primitives.Ints;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.ConstantDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Optional;
import java.util.stream.IntStream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link MutationGraph}
 *
 * @author Christian Skärby
 */
public class MutationGraphTest {

    private final static String inputName = "input";
    private final static String outputName = "output";

    private final static String W = DefaultParamInitializer.WEIGHT_KEY;
    private final static String B = DefaultParamInitializer.BIAS_KEY;


    /**
     * Test to change nOut in a CNN layer which is input to another CNN layer and see that weights get transferred
     */
    @Test
    public void changeNoutCnnToCnn() {
        final String mutationName = "toMutate";
        final String afterName = "afterMut";
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .weightInit(new ConstantDistribution(666))
                .graphBuilder()
                .addInputs(inputName)
                .setOutputs(outputName)
                .setInputTypes(InputType.convolutional(33, 33, 3))
                .addLayer(mutationName, new Convolution2D.Builder(1, 1)
                        .nOut(10)
                        .build(), inputName)
                .addLayer(afterName, new Convolution2D.Builder(1, 1)
                        .nOut(7)
                        .build(), mutationName)
                .addLayer(outputName, new OutputLayer.Builder()
                        .nOut(2)
                        .build(), afterName)
                .build());
        graph.init();

        setToLinspace(graph.getLayer(mutationName).getParam(W), true);
        setToLinspace(graph.getLayer(afterName).getParam(W), false);
        setToLinspace(graph.getLayer(mutationName).getParam(B), false);

        final int[] orderToKeep = {1, 3, 5, 6, 7, 9, 2, 4, 8, 0};
        final MutationGraph mutationGraph = new MutationGraph(graph,
                name -> name.equals(mutationName) ?
                        Optional.of(dummy -> (Comparator.comparingInt(i -> Ints.indexOf(orderToKeep, i)))) :
                        Optional.empty());

        final ComputationGraph newGraph = new TransferLearning
                .GraphBuilder(graph)
                .nOutReplace(mutationName, 5, new ConstantDistribution(333))
                .build();

        assertEquals("newGraph not initialized as expected!",
                333d,
                newGraph.getLayer(mutationName).paramTable().get(W).meanNumber().doubleValue(), 1e-10);

        assertEquals("newGraph not initialized as expected!",
                333d,
                newGraph.getLayer(afterName).paramTable().get(W).meanNumber().doubleValue(), 1e-10);


        final ComputationGraph mutatedGraph = mutationGraph.mutateTo(newGraph.getConfiguration());

        final INDArray source = graph.getLayer(mutationName).getParam(W);
        final INDArray target = mutatedGraph.getLayer(mutationName).getParam(W);
        assertDims(0, orderToKeep, source, target);

        final INDArray sourceBias = graph.getLayer(mutationName).getParam(B);
        final INDArray targetBias = mutatedGraph.getLayer(mutationName).getParam(B);
        assertDims(1, orderToKeep, sourceBias, targetBias);

        final INDArray sourceOutput = graph.getLayer(afterName).getParam(W);
        final INDArray targetOutput = mutatedGraph.getLayer(afterName).getParam(W);
        assertDims(1, orderToKeep, sourceOutput, targetOutput);
    }

    long[] assertDims(
            int dim,
            int[] orderToKeep,
            INDArray source,
            INDArray target) {
        final long[] shapeTarget = target.shape();
        final int[] dims = IntStream.range(0, shapeTarget.length).filter(i -> i!=dim).toArray();
        for (int elemInd = 0; elemInd < shapeTarget[dim]; elemInd++) {
            assertEquals("Incorrect target for element index " + elemInd + "!",
                    source.tensorAlongDimension(orderToKeep[elemInd], dims),
                    target.tensorAlongDimension(elemInd, dims));
        }
        return shapeTarget;
    }

    private static void setToLinspace(INDArray array, boolean reverse) {
        final long nrofElemsSource = Arrays.stream(array.shape()).reduce((i1, i2) -> i1 * i2).orElseThrow(() -> new IllegalArgumentException("No elements!"));
        array.assign(Nd4j.linspace(0, nrofElemsSource - 1, nrofElemsSource).reshape(array.shape()));
        if (reverse) {
            Nd4j.reverse(array);
        }
    }
}