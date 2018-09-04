package ampcontrol.model.training.model.mutate;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.junit.Test;

import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link MutateNout}
 *
 * @author Christian Skärby
 */
public class MutateNoutTest {

    /**
     * Test mutation function
     */
    @Test
    public void mutate() {
        final String mut1 = "mut1";
        final String mut2 = "mut2";
        final String noMut = "noMut";
        final ComputationGraph graph = MutationGraphTest.getCnnGraph(mut1, mut2, noMut);

        final MutateNout mutateNout = new MutateNout(() -> Stream.of(mut1, mut2), i -> 2*i);
        final ComputationGraph newGraph = mutateNout.mutate(new TransferLearning.GraphBuilder(graph), graph).build();

        assertEquals("Incorrect nOut!", 2 * getNout(graph.getLayer(mut1)), getNout(newGraph.getLayer(mut1)));
        assertEquals("Incorrect nOut!", 2 * getNout(graph.getLayer(mut2)), getNout(newGraph.getLayer(mut2)));
        assertEquals("Incorrect nOut!", getNout(graph.getLayer(noMut)), getNout(newGraph.getLayer(noMut)));
    }

    private long getNout(Layer layer) {
        return ((FeedForwardLayer)layer.conf().getLayer()).getNOut();
    }
}