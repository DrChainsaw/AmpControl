package ampcontrol.model.training.model.evolve.mutate;

import ampcontrol.model.training.model.evolve.GraphUtils;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Optional;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;

/**
 * Test cases for {@link MutateLayerContained}
 *
 * @author Christian Skärby
 */
public class MutateLayerContainedTest {

    /**
     * Test mutation by replacing two subsequent layers
     */
    @Test
    public void mutateReplace() {
        final String mut1 = "mut1";
        final String mut2 = "mut2";
        final String noMut = "noMut";
        final ComputationGraph graph = GraphUtils.getCnnGraph(mut1, mut2, noMut);

        final Mutation<TransferLearning.GraphBuilder>  mutatation = new MutateLayerContained(() -> Stream.of(
                MutateLayerContained.LayerMutation.builder()
                        .layerName(mut1)
                        .layerSupplier(() -> new Convolution2D.Builder(5, 5))
                        .inputLayers(getInputs(graph, mut1))
                        .build(),
                MutateLayerContained.LayerMutation.builder()
                        .layerName(mut2)
                        .layerSupplier(() -> new Convolution2D.Builder(7, 7))
                        .inputLayers(getInputs(graph, mut2))
                        .build()));
        final ComputationGraph newGraph = mutatation.mutate(new TransferLearning.GraphBuilder(graph)).build();
        newGraph.init();

        assertEquals("Incorrect kernel size!", 5,
                newGraph.getLayer(mut1).getParam(DefaultParamInitializer.WEIGHT_KEY).size(3));
        assertEquals("Incorrect kernel size", 7 ,
                newGraph.getLayer(mut2).getParam(DefaultParamInitializer.WEIGHT_KEY).size(2));
        assertEquals("Incorrect kernel size",
                graph.getLayer(noMut).getParam(DefaultParamInitializer.WEIGHT_KEY).size(3),
                newGraph.getLayer(noMut).getParam(DefaultParamInitializer.WEIGHT_KEY).size(3));

        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    /**
     * Test mutation by inserting a layer between two layers
     */
    @Test
    public void mutateInsert() {
        final String mut1 = "mut1";
        final String mut2 = "mut2";
        final String noMut = "noMut";
        final ComputationGraph graph = GraphUtils.getCnnGraph(mut1, mut2, noMut);

        final String toInsert = "between_" + mut1 + "_and_" + mut2;
        final Mutation<TransferLearning.GraphBuilder>  mutatation = new MutateLayerContained(() -> Stream.of(
                MutateLayerContained.LayerMutation.builder()
                        .layerName(toInsert)
                        .layerSupplier(() -> new Convolution2D.Builder(5, 5))
                        .inputLayers(new String[] {mut1})
                        .build()));
        final ComputationGraph newGraph = mutatation.mutate(new TransferLearning.GraphBuilder(graph)).build();
        newGraph.init();

        assertTrue("Vertex not added!", Optional.ofNullable(newGraph.getVertex(toInsert)).isPresent());
        assertEquals("Incorrect kernel size!", 5,
                newGraph.getLayer(toInsert).getParam(DefaultParamInitializer.WEIGHT_KEY).size(3));

        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    private String[] getInputs(ComputationGraph graph, String vertexName) {
        return graph.getConfiguration().getVertexInputs().get(vertexName).toArray(new String[] {});
    }
}