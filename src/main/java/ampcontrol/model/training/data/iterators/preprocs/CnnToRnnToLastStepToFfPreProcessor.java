package ampcontrol.model.training.data.iterators.preprocs;

import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

/**
 * {@link DataSetPreProcessor} which transforms CNN 4D features of a given {@link DataSet} to RNN 3D features. Typical
 * use case is when RNN output is to be sent to a LastTimeStep vertex.
 * <br><br>
 * Transforms features of shape [a,b,c,d] to shape [a,d,c] basically assuming that 1) input has only a single channel
 * (b == 1) and 2) c is the dimension which is relevant to view as time.
 *
 * @author Christian Skärby
 */
public class CnnToRnnToLastStepToFfPreProcessor implements DataSetPreProcessor {

    private final WorkspaceConfiguration workspaceConfig = WorkspaceConfiguration.builder()
            .initialSize(1024L*1024L)
            .policyAllocation(AllocationPolicy.STRICT)
            .policyLearning(LearningPolicy.FIRST_LOOP)
            .policyMirroring(MirroringPolicy.HOST_ONLY)
            .build();

    @Override
    public void preProcess(DataSet toPreProcess) {
        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, "CnnToRnnToLastStepToFfPreProcessorWs")) {
            toPreProcess.setFeatures(CnnToManyToOneRnnPreProcessor.cnnToRnnFeature(toPreProcess.getFeatures()));
        }
    }
}
