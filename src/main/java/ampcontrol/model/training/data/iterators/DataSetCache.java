package ampcontrol.model.training.data.iterators;

import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Cache for {@link DataSet}s. Cached sets will be stored in RAM until retrieved at which
 * point they are migrated (if part of a {@link MemoryWorkspace} scope) or detached.
 *
 * @author Christian Skärby
 */
class DataSetCache {

    private final String wsName = "DataSetCacheWs" + this.toString().split("@")[1];
    private final WorkspaceConfiguration cacheWorkspaceConfig = WorkspaceConfiguration.builder()
            .policyAllocation(AllocationPolicy.STRICT)
            .policyLearning(LearningPolicy.FIRST_LOOP)
            .policyMirroring(MirroringPolicy.HOST_ONLY)
            .policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
            .policySpill(SpillPolicy.REALLOCATE)
            .initialSize(0)
            //.overallocationLimit(20)
            .build();
    private List<DataSet> cache;
    private List<MemoryWorkspace> workspaces = new ArrayList<>();
    private int cursor = -1;
    private final boolean useWorkspace;


    /**
     * Constructor
     */
    DataSetCache() {
        this(!Nd4j.getBackend().getClass().getSimpleName().equals("CpuBackend"));
    }

    /**
     * Constructor
     *
     * @param useWorkspace True if a workspace shall be used (recommended for GPU operation).
     */
    DataSetCache(boolean useWorkspace) {
        this.useWorkspace = useWorkspace;
    }

    synchronized void check(DataSetIterator sourceIter, int nrofItersToCache) {
        if (cache != null) {
            return;
        }
        try (MemoryWorkspace outerWs = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            workspaces.stream().filter(Objects::nonNull).forEach(MemoryWorkspace::destroyWorkspace);
            workspaces.clear();

            cache = IntStream.range(0, nrofItersToCache)
                    .parallel()

                    .mapToObj(i -> {
                        if (useWorkspace) {
                            final MemoryWorkspace ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(cacheWorkspaceConfig, wsName + i);
                            workspaces.add(ws);
                            try (MemoryWorkspace wss = ws.notifyScopeEntered()) {
                                DataSet ds = sourceIter.next();
                                Nd4j.getExecutioner().commit();
                                return ds;
                            }
                        }
                        return sourceIter.next();
                    })
                    .collect(Collectors.toList());
        }
        resetCursor();
    }

    void clear() {
        cache = null;
    }

    void resetCursor() {
        cursor = -1;
    }

    boolean hasNext() {
        return isLoaded() && cache.size() - 1 > cursor;
    }

    boolean isLoaded() {
        return cache != null;
    }

    DataSet next() {
        if (!hasNext()) {
            throw new UnsupportedOperationException("Asked for next when no next exits!");
        }
        cursor++;
        return detach(cache.get(cursor));
    }

    private DataSet detach(DataSet ds) {
        if (ds == null) {
            return ds; // Happens in testing. CBA to change it
        }
        final INDArray features = detachOrMigrateIfNotNull(ds.getFeatures());
        final INDArray labels = detachOrMigrateIfNotNull(ds.getLabels());
        final INDArray featuresMask = detachOrMigrateIfNotNull(ds.getFeaturesMaskArray());
        final INDArray labelsMask = detachOrMigrateIfNotNull(ds.getLabelsMaskArray());
        return new DataSet(features, labels, featuresMask, labelsMask);
    }

    private INDArray detachOrMigrateIfNotNull(INDArray array) {
        if (array != null) {
            return array.migrate(true);
        }
        return null;
    }
}
