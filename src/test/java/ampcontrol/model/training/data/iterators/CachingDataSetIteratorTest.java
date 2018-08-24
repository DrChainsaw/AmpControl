package ampcontrol.model.training.data.iterators;

import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import static org.junit.Assert.*;

/**
 * Test cases for {@link CachingDataSetIterator}
 *
 * @author Christian Skärby
 */
public class CachingDataSetIteratorTest extends DecoratingMiniEpochIteratorTest {

    @Override
    protected MiniEpochDataSetIterator decorateMiniEpochIter(MiniEpochDataSetIterator anIterator) {
        return new CachingDataSetIterator(anIterator);
    }

    /**
     * Test that hasNext returns true only when cursor is reset
     */
    @Test
    public void hasNext() {
        final MiniEpochDataSetIterator iter = new CachingDataSetIterator(new MockDataSetIterator());
        assertTrue("Expected hasNext!", iter.hasNext());
        iter.next();
        assertFalse("Expected not hasNext!", iter.hasNext());
        iter.restartMiniEpoch();
        assertTrue("Expected hasNext!", iter.hasNext());
        iter.next();
        assertFalse("Expected not hasNext!", iter.hasNext());
    }

    /**
     * Test that next, restartMiniEpoch and reset works as expected: next creates the cache if not initialized, restartMiniEpoch
     * produces the same sequence of data sets when calling again while reset creates a new sequence of data sets.
     */
    @Test
    public void nextAndReset() {
        final int cacheSize = 3;
        final DataSet ds0 = new DataSet();
        final DataSet ds1 = new DataSet();
        final DataSetIterator sourceMock = new MockDataSetIterator() {
            private boolean toggle = false;

            @Override
            public DataSet next() {
                toggle = !toggle;
                return toggle ? ds0 : ds1;
            }
        };
        final MiniEpochDataSetIterator iter = new CachingDataSetIterator(sourceMock, cacheSize);
        assertEquals("Incorrect Data set!", ds0, iter.next());
        assertEquals("Incorrect Data set!", ds1, iter.next());
        assertEquals("Incorrect Data set!", ds0, iter.next());
        iter.restartMiniEpoch();
        assertEquals("Incorrect Data set!", ds0, iter.next());
        assertEquals("Incorrect Data set!", ds1, iter.next());
        assertEquals("Incorrect Data set!", ds0, iter.next());
        iter.reset();
        assertEquals("Incorrect Data set!", ds1, iter.next());
        assertEquals("Incorrect Data set!", ds0, iter.next());
        assertEquals("Incorrect Data set!", ds1, iter.next());
        iter.restartMiniEpoch();
        assertEquals("Incorrect Data set!", ds1, iter.next());
        assertEquals("Incorrect Data set!", ds0, iter.next());
        assertEquals("Incorrect Data set!", ds1, iter.next());
    }

    /**
     * Test handling of {@link DataSetPreProcessor DataSetPreProcessors}: If cache is not initialized then set the
     * pre-processor of the sourceIter and do not do pre-processing. If initialized but source and cache have the same
     * pre-processor instance then do not perform pre-processing. If source has no pre-processoer then perform
     * pre-processing.
     */
    @Test
    public void setPreProcessor() {
        final ProbingPreProcessor preProcessor = new ProbingPreProcessor();
        final DataSetIterator sourceMock = new MockDataSetIterator() {
            @Override
            public DataSet next() {
                return new DataSet();
            }
        };
        final DataSetIterator iter = new CachingDataSetIterator(sourceMock, 2);
        iter.setPreProcessor(preProcessor);
        assertEquals("Incorrect preProcessor!", preProcessor, sourceMock.getPreProcessor());
        iter.next();
        // Expect that source will call pre-processor in normal case, but MockDataSetIterator won't
        preProcessor.assertWasCalled(false);
        iter.setPreProcessor(preProcessor);
        iter.next();
        // This time around CachingDataSetIterator will not call pre-processor as cache is already created
        preProcessor.assertWasCalled(false);
        iter.reset();
        sourceMock.setPreProcessor(null);
        iter.next();
        // This time around CachingDataSetIterator will call pre-processor as source has no pre-processor
        preProcessor.assertWasCalled(true);
    }


    /**
     * Test that a {@link RuntimeException} is thrown when source and cache have different pre-processor instances.
     */
    @Test(expected = RuntimeException.class)
    public void setPreProcessorDifferent() {
        final DataSetIterator sourceMock = new MockDataSetIterator();
        final MiniEpochDataSetIterator iter = new CachingDataSetIterator(sourceMock);
        sourceMock.setPreProcessor(new ProbingPreProcessor());
        iter.next(); // So that cache is created
        iter.restartMiniEpoch(); // Does not reset cache
        iter.setPreProcessor(new ProbingPreProcessor()); // Now preprocessor instances are different in source and cache
        iter.next(); // Exception!
    }

    @Override
    public void resetSupported() {
        assertTrue("Reset shall be supported!", decorate(new MockMiniEpochDataSetIterator()).resetSupported());
    }

    @Override
    public void restartMiniEpoch() {
        // Not supposed to propagate
    }

    private static final class ProbingPreProcessor implements DataSetPreProcessor {

        private boolean wasCalled = false;

        @Override
        public void preProcess(org.nd4j.linalg.dataset.api.DataSet toPreProcess) {
            wasCalled = true;
        }

        private void assertWasCalled(boolean expected) {
            assertEquals("Incorrec preprocessor invocation!", expected, wasCalled);
        }
    }
}