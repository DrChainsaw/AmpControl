package ampcontrol.model.training.model.evolve.crossover.graph;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.util.CompGraphUtil;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Test cases for {@link CrossoverPoint}
 *
 * @author Christian Skärby
 */
public class CrossoverPointTest {

    /**
     * Test to do crossover when the vertex name is identical
     */
    @Test
    public void executeSameVertexName() {
        final ComputationGraphConfiguration.GraphBuilder builder1 = CompGraphUtil.toBuilder(GraphUtils.getGraph("0", "1", "2"))
                .setInputTypes(InputType.feedForward(33));

        final ComputationGraphConfiguration.GraphBuilder builder2 = CompGraphUtil.toBuilder(GraphUtils.getGraph("0", "1", "2"))
                .setInputTypes(InputType.feedForward(33));

        GraphInfo result = new CrossoverPoint(
                new VertexData("1", new GraphInfo.Input(builder1)),
                new VertexData("1", new GraphInfo.Input(builder2)))
                .execute();

        final ComputationGraph graph = new ComputationGraph(result.builder().build());
        graph.init();
        graph.output(Nd4j.randn(new long[] {1, 33}));
    }
}