package ampcontrol.model.training.model.evolve.mutate;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.transfer.ParameterTransfer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.junit.Test;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.factory.Nd4j;

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
        final ComputationGraph graph = GraphUtils.getCnnGraph(mut1, mut2, noMut);

        final MutateNout mutateNout = new MutateNout(() -> Stream.of(mut1, mut2), i -> 2 * i);
        final ComputationGraph newGraph = mutateNout.mutate(new TransferLearning.GraphBuilder(graph), graph).build();
        newGraph.init();

        assertEquals("Incorrect nOut!", 2 * graph.layerSize(mut1), newGraph.layerSize(mut1));
        assertEquals("Incorrect nOut!", 2 * graph.layerSize(mut2), newGraph.layerSize(mut2));
        assertEquals("Incorrect nOut!", graph.layerSize(noMut), newGraph.layerSize(noMut));

        graph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
        newGraph.outputSingle(Nd4j.randn(new long[]{1, 3, 33, 33}));
    }

    public static void main(String[] args) {
        int cnt = 0;
        final String inputName = "input";
        final String outputName = "output";
        final String conv1Name = "conv1";
        final String conv2Name = "conv2";
        final int conv1Nout = 1000;
        ComputationGraph graph = GraphUtils.getNewGraph(inputName, outputName, conv1Name, conv2Name, conv1Nout);

        final MemoryWorkspace workspace = Nd4j.getWorkspaceManager().createNewWorkspace(WorkspaceConfiguration.builder()
                        .policyAllocation(AllocationPolicy.STRICT)
                        .policyLearning(LearningPolicy.FIRST_LOOP)
                        .policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                        .policySpill(SpillPolicy.REALLOCATE)
                        .initialSize(0)
                        .build(),
                "testWorkspace");
        while (true) {
            final Mutation mutateNout = new MutateNout(() -> Stream.of(conv1Name), i -> i > 20 ? i-1 : conv1Nout);
            try(MemoryWorkspace ws = workspace.notifyScopeEntered()) {
                ComputationGraph newGraph = mutateNout.mutate(new TransferLearning.GraphBuilder(graph), graph).build();
               // graph = newGraph;
                graph = new ParameterTransfer(graph).transferWeightsTo(newGraph);
            }


            System.out.println(cnt + " npar : " + graph.numParams());
            cnt++;
        }

    }
}