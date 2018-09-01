package ampcontrol.model.training.data.iterators.factory;

import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import ampcontrol.model.training.data.state.ResetableState;
import lombok.Builder;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.stream.IntStream;

/**
 * Tries to automatically select and appropriate {@link MiniEpochDataSetIterator} based on memory restrictions.
 *
 * @param <V> Type of input to source factory
 * @author Christian Skärby
 */
public class AutoFromSize<V> implements DataSetIteratorFactory<MiniEpochDataSetIterator, AutoFromSize.Input<V>> {

    private static final Logger log = LoggerFactory.getLogger(AutoFromSize.class);
    private static final double margin = 1.2; // How much margin to use when calculating memory size of data sets

    private long memoryAllowance;
    private final long dataTypeSize;

    @Builder
    public static class Input<V> {
        private final DataSetIteratorFactory<?, V> sourceFactory;
        private final V sourceInput;
        private final int dataSetSize;
        private final int[] dataSetShape;
        private final int batchSize;
        private final ResetableState resetableState;
    }

    /**
     * Constructor. Will try to figure out memory restrictions and costs by itself
     */
    public AutoFromSize() {
        this(figureOutMemLimit(), figureOutDataTypeSize());
    }

    /**
     * Constructor. Will try to figure out memory costs by itself
     * @param memoryAllowance How much memory data sets are allowed to use
     */
    public AutoFromSize(long memoryAllowance) {
        this(memoryAllowance, figureOutDataTypeSize());
    }

    /**
     * Constructor
     * @param memoryAllowance How much memory data sets are allowed to use
     * @param dataTypeSize Cost for a single variable (e.g. one element in a feature vector/matrix)
     */
    public AutoFromSize(long memoryAllowance, long dataTypeSize) {
        this.memoryAllowance = memoryAllowance;
        this.dataTypeSize = dataTypeSize;
    }

    private static long figureOutMemLimit() {
        return Optional.ofNullable(System.getProperty("org.bytedeco.javacpp.maxbytes"))
                .map(str -> str.replaceAll("G", String.valueOf(1024L * 1024L * 1024L))) // CBA to do M and K (and T)
                .map(Long::valueOf)
                // Also check vs RAM in case maxMemory * 2 is more than system RAM
                .orElse(Runtime.getRuntime().maxMemory() * 2);
    }

    private static long figureOutDataTypeSize() {
        return DataTypeUtil.lengthForDtype(DataTypeUtil.getDtypeFromContext());
    }

    @Override
    public MiniEpochDataSetIterator create(Input<V> input) {

        final long sizeOfOneBatch = (IntStream.of(input.dataSetShape).reduce(1, (i1,i2) -> i1*i2)) * input.batchSize * dataTypeSize;
        final long sizeOfWholeDataSet = sizeOfOneBatch * input.dataSetSize;
        DataSetIteratorFactory<MiniEpochDataSetIterator, V> factory = createFactory(input, sizeOfOneBatch, sizeOfWholeDataSet);

        return factory.create(input.sourceInput);
    }

    @NotNull
    private DataSetIteratorFactory<MiniEpochDataSetIterator, V> createFactory(Input<V> input, long sizeOfOneBatch, long sizeOfWholeDataSet) {
        DataSetIteratorFactory<MiniEpochDataSetIterator, V> factory;
        if (margin * sizeOfWholeDataSet > memoryAllowance) {
            factory = createAsynchFactory(input, sizeOfOneBatch, sizeOfWholeDataSet);
        } else {
            factory = createCachingFactory(input, sizeOfWholeDataSet);
        }
        return factory;
    }

    @NotNull
    private DataSetIteratorFactory<MiniEpochDataSetIterator, V> createAsynchFactory(Input<V> input, long sizeOfOneBatch, long sizeOfWholeDataSet) {
        DataSetIteratorFactory<MiniEpochDataSetIterator, V> factory;
        log.info("Create Asynch iter for set of size {} with memory allowance {}", sizeOfWholeDataSet, memoryAllowance);
        final int bufferSize = Math.max(4, Runtime.getRuntime().availableProcessors() / 3);
        final int miniEpochSizeTrunc = (input.dataSetSize / bufferSize) * bufferSize;
        log.info("Data set size changed from {} to {} in order to align buffer", input.dataSetSize, miniEpochSizeTrunc);
        factory = new Asynch<>(miniEpochSizeTrunc, input.resetableState, new DoubleBuffered<>(bufferSize, input.sourceFactory));
        memoryAllowance -= bufferSize * 2 * sizeOfOneBatch;
        return factory;
    }

    @NotNull
    private DataSetIteratorFactory<MiniEpochDataSetIterator, V> createCachingFactory(Input<V> input, long sizeOfWholeDataSet) {
        DataSetIteratorFactory<MiniEpochDataSetIterator, V> factory;
        log.info("Create Caching iter for set of size {} with memory allowance {}", sizeOfWholeDataSet, memoryAllowance);
        factory = new WorkSpaceWrapping<>(new Cached<>(input.dataSetSize, input.sourceFactory));
        memoryAllowance -= sizeOfWholeDataSet;
        return factory;
    }
}
