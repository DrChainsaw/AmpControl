package ampcontrol.model.training.data.iterators;


import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * {@link DataSetIterator} which caches output from another {@link DataSetIterator}. Main use case is validation set but
 * also useful for training multiple models in parallell using the same training data.
 *
 * @author Christian Skärby
 */
public class CachingDataSetIterator implements DataSetIterator {
    private static final Logger log = LoggerFactory.getLogger(CachingDataSetIterator.class);

    /**
     *
     */
    private static final long serialVersionUID = 6874213288810185979L;
    private final DataSetIterator sourceIter;
    private final int nrofItersToCache;
    private final boolean useWorkspace;
    private List<DataSet> cache;
    private int cursor = -1;

    private DataSetPreProcessor preProcessor;

    private final WorkspaceConfiguration workspaceConfig = WorkspaceConfiguration.builder()
            .policyAllocation(AllocationPolicy.STRICT)
            .policyLearning(LearningPolicy.FIRST_LOOP)
            .policyMirroring(MirroringPolicy.HOST_ONLY)
            .build();


    /**
     * Constructor
     *
     * @param sourceIter {@link DataSetIterator} for which a cache shall be created
     */
    public CachingDataSetIterator(DataSetIterator sourceIter) {
        this(sourceIter, 1);
    }

    /**
     * Constructor
     *
     * @param sourceIter       {@link DataSetIterator} for which a cache shall be created
     * @param nrofItersToCache Sets how many iterations from sourceIter will be cached.
     */
    public CachingDataSetIterator(DataSetIterator sourceIter, int nrofItersToCache) {
        this.sourceIter = sourceIter;
        this.nrofItersToCache = nrofItersToCache;
        useWorkspace = !Nd4j.getBackend().getClass().getSimpleName().equals("CpuBackend");
    }

    @Override
    public boolean hasNext() {
        return cursor + 1 < nrofItersToCache;
    }

    @Override
    public DataSet next() {
        if (cache == null) {
            log.info("create cache of size " + nrofItersToCache);
            cache = IntStream.range(0, nrofItersToCache)
                    .parallel()

                    .mapToObj(i -> {
                        if(useWorkspace) {
                            // Create a new workspace for each thread started or else data becomes all zeroes
                            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, "CachingDataSetWs" + i)) {
                                return migrate(sourceIter.next());
                            }
                        }
                        return sourceIter.next();

                    })
                    .collect(Collectors.toList());

            resetCursor();
        }
        cursor++;
        DataSet ds = cache.get(cursor);

        // Handle pre-processing of data. There can only be one type of PreProcessor between this class and the sourceIter
        if (preProcessor != null) {
            if (sourceIter.getPreProcessor() == null) {
                ds = new DataSet(ds.getFeatures(), ds.getLabels(), ds.getFeaturesMaskArray(), ds.getLabelsMaskArray());
                preProcessor.preProcess(ds);
            } else if (sourceIter.getPreProcessor() != preProcessor) {
                throw new IllegalStateException("Different preprocessors for source and cache! Source: "
                        + sourceIter.getPreProcessor() + " cache: " + preProcessor);
            }
        }
        return ds;
    }

    private DataSet migrate(DataSet ds) {
        if (ds == null) { // Happens in testing. CBA to change it
            return ds;
        }
        ds.detach();
        return ds;
    }

    @Override
    public DataSet next(int num) {
        throw new UnsupportedOperationException("Not supported!");
    }

    @Override
    public int totalExamples() {
        return sourceIter.totalExamples();
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
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return sourceIter.asyncSupported();
    }

    @Override
    public void reset() {
        cache = null;
        if (sourceIter.resetSupported()) {
            sourceIter.reset();
        }
    }

    @Override
    public int batch() {
        return sourceIter.batch();
    }

    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        return sourceIter.numExamples() * nrofItersToCache;
    }

    /**
     * Sets the DataSetPreProcessor to use. In case this happens before the cache has been created the PreProcessor will
     * be set for the sourceIter so that the cache will consist of pre-processed data. Otherwise the {@link DataSet DataSets}
     * in the cache will be pre-processed instead.
     * <br><br>
     * Beware that in case the sourceIter has another PreProcessor than this class an exception will be thrown. Therefore
     * one must only call this method after cache has been created when training different models with different
     * pre-processing.
     *
     * @param preProcessor the preprocessor to use
     */
    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
        if (cache == null) {
            sourceIter.setPreProcessor(preProcessor);
        }
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return sourceIter.getPreProcessor();
    }

    @Override
    public List<String> getLabels() {
        return sourceIter.getLabels();
    }

    /**
     * Resets the cursor for the cache so that the same {@link DataSet DataSets} will be provided again
     */
    public void resetCursor() {
        cursor = -1;
    }

}
