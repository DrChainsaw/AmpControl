package ampcontrol.model.training.model.evolve.mutate.layer;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.Mutation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link LayerContainedMutation}
 *
 * @author Christian Skärby
 */
public class LayerContainedMutationTest {

    /**
     * Test mutation by replacing two subsequent layers
     */
    @Test
    public void mutateReplace() {
        final String mut1 = "mut1";
        final String mut2 = "mut2";
        final String noMut = "noMut";
        final ComputationGraph graph = GraphUtils.getCnnGraph(mut1, mut2, noMut);

        final Mutation<ComputationGraphConfiguration.GraphBuilder> mutatation = new LayerContainedMutation(() -> Stream.of(
                LayerContainedMutation.LayerMutation.builder()
                        .mutationInfo(
                                LayerMutationInfo.builder()
                                        .layerName(mut1)
                                        .build())
                        .mutation(layer -> new Convolution2D.Builder(5, 5).build())
                        .build(),
                LayerContainedMutation.LayerMutation.builder()
                        .mutationInfo(
                                LayerMutationInfo.builder()
                                        .layerName(mut2)
                                        .build())
                        .mutation(layer -> new Convolution2D.Builder(7, 7).build())
                        .build()));
        final ComputationGraph newGraph = new ComputationGraph(mutatation.mutate(
                new ComputationGraphConfiguration.GraphBuilder(graph.getConfiguration(),
                        new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        assertEquals("Incorrect kernel size!", 5,
                newGraph.getLayer(mut1).getParam(DefaultParamInitializer.WEIGHT_KEY).size(3));
        assertEquals("Incorrect kernel size", 7,
                newGraph.getLayer(mut2).getParam(DefaultParamInitializer.WEIGHT_KEY).size(2));
        assertEquals("Incorrect kernel size",
                graph.getLayer(noMut).getParam(DefaultParamInitializer.WEIGHT_KEY).size(3),
                newGraph.getLayer(noMut).getParam(DefaultParamInitializer.WEIGHT_KEY).size(3));

        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }
}