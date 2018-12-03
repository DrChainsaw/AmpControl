package ampcontrol.model.training.model.evolve.mutate.util;

import ampcontrol.model.training.model.evolve.GraphUtils;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for various usages of {@link Graph}
 *
 * @author Christian Skärby
 */
public class GraphUtilTest {


    /**
     * Test traversal to find the minimum Nout of a simple {@link ComputationGraphConfiguration.GraphBuilder}.
     */
    @Test
    public void testFindMinNoutInSimpleGraph() {
        final String toQuery = "mid";
        final long expected = 0;
        final ComputationGraph computationGraph = GraphUtils.getCnnGraph("first", toQuery,"last");
        assertMinNout(toQuery, expected, computationGraph);
    }

    /**
     * Test traversal to find the minimum Nout of a {@link ComputationGraphConfiguration.GraphBuilder} with a
     * double residual forked which is in turn forked.
     */
    @Test
    public void testFindMinNoutInFirstFork() {
        final String toQuery = "before";
        final String[] forkNames = {"f1", "f2", "f3", "f4", "f5"};
        final long expected = 5;
        final ComputationGraph computationGraph = GraphUtils.getDoubleForkResNet(toQuery, "after", forkNames);
        assertMinNout(toQuery, expected, computationGraph);
    }

    /**
     * Test traversal to find the minimum Nout of a {@link ComputationGraphConfiguration.GraphBuilder} with a
     * double residual forked which is in turn forked.
     */
    @Test
    public void testFindMinNoutInSecondFork() {
        final String toQuery = "before";
        final String[] forkNames = {"f1", "f2"};
        final long expected = 4;
        final ComputationGraph computationGraph = GraphUtils.getDoubleForkResNet(toQuery, "after", forkNames);
        assertMinNout(toQuery, expected, computationGraph);
    }

    private static void assertMinNout(String toQuery, long expected, ComputationGraph computationGraph) {
        final ComputationGraphConfiguration.GraphBuilder builder = new ComputationGraphConfiguration.GraphBuilder(
                computationGraph.getConfiguration(), new NeuralNetConfiguration.Builder(computationGraph.conf()));

        final Graph<String> forwardGraph = TraverseBuilder.forwards(builder).build();
        final Graph<String> backwardGraph =
                new Filter<>(GraphBuilderUtil.changeSizePropagates(builder).negate(),
                        TraverseBuilder.backwards(builder)
                                .visitCondition(vertex -> !vertex.equals(toQuery))
                                .build());
        final long minNout = forwardGraph.children(toQuery)
                .mapToLong(childName -> backwardGraph.children(childName).count())
                .max().orElse(0);
        assertEquals("Incorrect minNout!!", expected, minNout);
    }
}