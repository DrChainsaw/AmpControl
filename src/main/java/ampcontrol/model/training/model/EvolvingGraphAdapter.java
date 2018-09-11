package ampcontrol.model.training.model;

import ampcontrol.model.training.model.evolve.Evolving;
import ampcontrol.model.training.model.evolve.mutate.Mutation;
import ampcontrol.model.training.model.evolve.transfer.ParameterTransfer;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * {@link ModelAdapter} which may evolve through a {@link Mutation}.
 *
 * @author Christian Skärby
 */
public class EvolvingGraphAdapter implements ModelAdapter, Evolving<EvolvingGraphAdapter> {

    private static final Logger log = LoggerFactory.getLogger(EvolvingGraphAdapter.class);

    private final ComputationGraph graph;
    private final Mutation mutation;

    final MemoryWorkspace workspace = Nd4j.getWorkspaceManager().createNewWorkspace(WorkspaceConfiguration.builder()
                    .policyAllocation(AllocationPolicy.OVERALLOCATE)
            .overallocationLimit(1.2)
                    .policyLearning(LearningPolicy.OVER_TIME)
                    .policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                    .policySpill(SpillPolicy.REALLOCATE)
                    .initialSize(0)
                    .build(),
            this.getClass().getSimpleName() + "Workspace" + this.toString().split("@")[1]);

    public EvolvingGraphAdapter(ComputationGraph graph, Mutation mutation) {
        this.graph = graph;
        this.mutation = mutation;
    }

    @Override
    public void fit(DataSetIterator iter) {
        graph.fit(iter);
    }

    @Override
    public <T extends IEvaluation> T[] eval(DataSetIterator iter, T... evals) {
        return graph.doEvaluation(iter, evals);
    }

    @Override
    public Model asModel() {
        return graph;
    }

    /**
     * Evolve the graph adapter
     * @return the evolved adapter
     */
    @Override
    public synchronized EvolvingGraphAdapter evolve() {
        try(MemoryWorkspace ws = workspace.notifyScopeEntered()) {
            log.info("Evolve " + this);
            graph.getListeners().clear();
            final TransferLearning.GraphBuilder mutated = mutation.mutate(new TransferLearning.GraphBuilder(graph), graph);
            final ParameterTransfer parameterTransfer = new ParameterTransfer(graph);
            final ComputationGraph newGraph = mutated.build();
            newGraph.getConfiguration().setIterationCount(graph.getIterationCount());
            newGraph.getConfiguration().setEpochCount(graph.getEpochCount());
            return new EvolvingGraphAdapter(parameterTransfer.transferWeightsTo(newGraph), mutation);
        }
    }
}
