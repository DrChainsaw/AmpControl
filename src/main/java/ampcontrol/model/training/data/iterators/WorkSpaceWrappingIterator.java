package ampcontrol.model.training.data.iterators;

import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * Puts the {@link #next()} method of a given {@link MiniEpochDataSetIterator} in a workspace scope.
 *
 * @author Christian Skärby
 */
public class WorkSpaceWrappingIterator implements MiniEpochDataSetIterator {

    private final MiniEpochDataSetIterator sourceIter;

    private final WorkspaceConfiguration workspaceConfig = WorkspaceConfiguration.builder()
            .policyAllocation(AllocationPolicy.STRICT)
            .policyLearning(LearningPolicy.FIRST_LOOP)
            .policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
            .policySpill(SpillPolicy.REALLOCATE)
            .initialSize(0)
            .build();

    private final String wsName = "WorkspaceWrappingIteratorWs" + this.toString().split("@")[1];

    public WorkSpaceWrappingIterator(MiniEpochDataSetIterator sourceIter) {
        this.sourceIter = sourceIter;
    }

    @Override
    public int inputColumns() {
        return sourceIter.inputColumns();
    }

    @Override
    public int totalOutcomes() {
        return sourceIter.totalOutcomes();
    }

    @Override
    public boolean resetSupported() {
        return sourceIter.resetSupported();
    }

    @Override
    public boolean asyncSupported() {
        return sourceIter.asyncSupported();
    }

    @Override
    public void reset() {
        sourceIter.reset();
    }

    @Override
    public int batch() {
        return sourceIter.batch();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        sourceIter.setPreProcessor(preProcessor);
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return sourceIter.getPreProcessor();
    }

    @Override
    public List<String> getLabels() {
        return sourceIter.getLabels();
    }

    @Override
    public boolean hasNext() {
        return sourceIter.hasNext();
    }

    @Override
    public DataSet next() {
        // Move to workspace for processing
        final MemoryWorkspace tmpWs = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceConfig, wsName);
        try (MemoryWorkspace ws = tmpWs.notifyScopeEntered()) {
            return sourceIter.next();
        }
    }

    @Override
    public void restartMiniEpoch() {
        sourceIter.restartMiniEpoch();
    }

    @Override
    public int miniEpochSize() {
        return sourceIter.miniEpochSize();
    }
}
