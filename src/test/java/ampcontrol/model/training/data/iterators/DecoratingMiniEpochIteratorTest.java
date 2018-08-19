package ampcontrol.model.training.data.iterators;

import org.deeplearning4j.datasets.iterator.DummyPreProcessor;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertTrue;

/**
 * Base test for {@link MiniEpochDataSetIterator} which decorates other iterators. By default it checks that the
 * underlying {@link MiniEpochDataSetIterator} is called. Override methods should this not be the case. I can certainly
 * see this class lifted up in a unit test bashing blog :)
 *
 * @author Christian Skärby
 */
public abstract class DecoratingMiniEpochIteratorTest {

    /**
     * Factory method for the instance to test
     * @param anIterator iterator to decorate
     * @return the iterator to test
     */
    protected abstract MiniEpochDataSetIterator decorate(MiniEpochDataSetIterator anIterator);

    /**
     * Test that the method is called from the decorating instance
     */
    @Test
    public void next() {
        final boolean[] wasCalled = {false};
        decorate(new MockMiniEpochDataSetIterator() {
            @Override
            public DataSet next() {
                wasCalled[0] = true;
                return new DataSet();
            }
        }).next();
        assertTrue("Method was not called!", wasCalled[0]);
    }

    /**
     * Test that the method is called from the decorating instance
     */
    @Test
    public void inputColumns() {
        final boolean[] wasCalled = {false};
        decorate(new MockMiniEpochDataSetIterator() {
            @Override
            public int inputColumns() {
                wasCalled[0] = true;
                return 0;
            }
        }).inputColumns();
        assertTrue("Method was not called!", wasCalled[0]);
    }

    /**
     * Test that the method is called from the decorating instance
     */
    @Test
    public void totalOutcomes() {
        final boolean[] wasCalled = {false};
        decorate(new MockMiniEpochDataSetIterator() {
            @Override
            public int totalOutcomes() {
                wasCalled[0] = true;
                return 0;
            }
        }).totalOutcomes();
        assertTrue("Method was not called!", wasCalled[0]);
    }

    /**
     * Test that the method is called from the decorating instance
     */
    @Test
    public void resetSupported() {
        final boolean[] wasCalled = {false};
        decorate(new MockMiniEpochDataSetIterator() {
            @Override
            public boolean resetSupported() {
                wasCalled[0] = true;
                return false;
            }
        }).resetSupported();
        assertTrue("Method was not called!", wasCalled[0]);
    }

    /**
     * Test that the method is called from the decorating instance
     */
    @Test
    public void asyncSupported() {
        final boolean[] wasCalled = {false};
        decorate(new MockMiniEpochDataSetIterator() {
            @Override
            public boolean asyncSupported() {
                wasCalled[0] = true;
                return false;
            }
        }).asyncSupported();
        assertTrue("Method was not called!", wasCalled[0]);
    }

    /**
     * Test that the method is called from the decorating instance
     */
    @Test
    public void reset() {
        final boolean[] wasCalled = {false};
        decorate(new MockMiniEpochDataSetIterator() {
            @Override
            public boolean hasNext() {
                return false;
            }

            @Override
            public void reset() {
                wasCalled[0] = true;
            }
        }).reset();
        assertTrue("Method not was called!", wasCalled[0]);
    }

    /**
     * Test that the method is called from the decorating instance
     */
    @Test
    public void batch() {
        final boolean[] wasCalled = {false};
        decorate(new MockMiniEpochDataSetIterator() {
            @Override
            public int batch() {
                wasCalled[0] = true;
                return 0;
            }
        }).batch();
        assertTrue("Method was not called!", wasCalled[0]);
    }

    /**
     * Test that the method is called from the decorating instance
     */
    @Test
    public void setPreProcessor() {
        final boolean[] wasCalled = {false};
        decorate(new MockMiniEpochDataSetIterator() {
            @Override
            public void setPreProcessor(DataSetPreProcessor preProcessor) {
                wasCalled[0] = true;
            }
        }).setPreProcessor(new DummyPreProcessor());
        assertTrue("Method was not called!", wasCalled[0]);
    }

    /**
     * Test that the method is called from the decorating instance
     */
    @Test
    public void getPreProcessor() {
        final boolean[] wasCalled = {false};
        decorate(new MockMiniEpochDataSetIterator() {
            @Override
            public DataSetPreProcessor getPreProcessor() {
                wasCalled[0] = true;
                return new DummyPreProcessor();
            }
        }).getPreProcessor();
        assertTrue("Method was not called!", wasCalled[0]);
    }

    /**
     * Test that the method is called from the decorating instance
     */
    @Test
    public void getLabels() {
        final boolean[] wasCalled = {false};
        decorate(new MockMiniEpochDataSetIterator() {
            @Override
            public List<String> getLabels() {
                wasCalled[0] = true;
                return Collections.emptyList();
            }
        }).getLabels();
        assertTrue("Method was not called!", wasCalled[0]);
    }

    /**
     * Test that the method is called from the decorating instance
     */
    @Test
    public void hasNext() {
        final boolean[] wasCalled = {false};
        decorate(new MockMiniEpochDataSetIterator() {
            @Override
            public boolean asyncSupported() {
                wasCalled[0] = true;
                return false;
            }
        }).asyncSupported();
        assertTrue("Method was not called!", wasCalled[0]);
    }

    /**
     * Test that the method is called from the decorating instance
     */
    @Test
    public void restartMiniEpoch() {
        final boolean[] wasCalled = {false};
        decorate(new MockMiniEpochDataSetIterator() {
            @Override
            public void restartMiniEpoch() {
                wasCalled[0] = true;
            }
        }).restartMiniEpoch();
        assertTrue("Method was not called!", wasCalled[0]);
    }
}