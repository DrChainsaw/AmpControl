package ampcontrol.model.training.data.iterators;

import org.junit.Test;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import static org.junit.Assert.assertTrue;

/**
 * Base test for {@link MiniEpochDataSetIterator} which decorates other iterators. By default it checks that the
 * underlying {@link MiniEpochDataSetIterator} is called. Override methods should this not be the case. I can certainly
 * see this class lifted up in a unit test bashing blog :)
 *
 * @author Christian Skärby
 */
public abstract class DecoratingMiniEpochIteratorTest extends DecoratingDataSetIteratorTest {

    /**
     * Factory method for the instance to test
     * @param sourceIter iterator to decorate
     * @return the iterator to test
     */
    protected abstract MiniEpochDataSetIterator decorateMiniEpochIter(MiniEpochDataSetIterator sourceIter);

    @Override
    protected final DataSetIterator decorate(MiniEpochDataSetIterator sourceIter) {
        return decorateMiniEpochIter(sourceIter);
    }

    /**
     * Test that the method is called from the decorating instance
     */
    @Test
    public void restartMiniEpoch() {
        final boolean[] wasCalled = {false};
        decorateMiniEpochIter(new MockMiniEpochDataSetIterator() {
            @Override
            public void restartMiniEpoch() {
                wasCalled[0] = true;
            }
        }).restartMiniEpoch();
        assertTrue("Method was not called!", wasCalled[0]);
    }

    /**
     * Test that operation is not supported
     */
    @Test(expected = UnsupportedOperationException.class)
    public void nextNum() {
        decorateMiniEpochIter(new MockMiniEpochDataSetIterator()).next(1);
    }
}