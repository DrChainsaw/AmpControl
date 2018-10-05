package ampcontrol.model.training.model.evolve.mutate;

import ampcontrol.model.training.model.evolve.GraphUtils;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link NoutMutation}
 *
 * @author Christian Skärby
 */
public class NoutMutationTest {

    /**
     * Test mutation function
     */
    @Test
    public void mutate() {
        final String mut1 = "mut1";
        final String mut2 = "mut2";
        final String noMut = "noMut";
        final ComputationGraph graph = GraphUtils.getCnnGraph(mut1, mut2, noMut);

        final NoutMutation noutMutation = new NoutMutation(() -> Stream.of(
                NoutMutation.NoutMutationDescription.builder().layerName(mut1).mutateNout(nOut -> 2 * nOut).build(),
                NoutMutation.NoutMutationDescription.builder().layerName(mut2).mutateNout(nOut -> 2 * nOut).build()));
        final ComputationGraph newGraph = new ComputationGraph(noutMutation.mutate(
                new ComputationGraphConfiguration.GraphBuilder(graph.getConfiguration(), new NeuralNetConfiguration.Builder(graph.conf())))
                .build());
        newGraph.init();

        assertEquals("Incorrect nOut!", 2 * graph.layerSize(mut1), newGraph.layerSize(mut1));
        assertEquals("Incorrect nOut!", 2 * graph.layerSize(mut2), newGraph.layerSize(mut2));
        assertEquals("Incorrect nOut!", graph.layerSize(noMut), newGraph.layerSize(noMut));

        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }
    }