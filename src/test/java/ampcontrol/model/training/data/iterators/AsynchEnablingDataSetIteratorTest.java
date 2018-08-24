package ampcontrol.model.training.data.iterators;

import ampcontrol.model.training.data.state.ResetableReferenceState;
import ampcontrol.model.training.data.state.ResetableStateFactory;
import org.apache.commons.lang.mutable.MutableInt;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.stream.IntStream;

import static org.junit.Assert.*;

/**
 * Test cases for {@link AsynchEnablingDataSetIterator}
 *
 * @author Christian Skärby
 */
public class AsynchEnablingDataSetIteratorTest extends DecoratingMiniEpochIteratorTest {


    @Override
    protected MiniEpochDataSetIterator decorateMiniEpochIter(MiniEpochDataSetIterator anIterator) {
        return new AsynchEnablingDataSetIterator(anIterator, new ResetableStateFactory(123), 1);
    }

    @Override
    public void restartMiniEpoch() {
        final ResetableReferenceState<MutableInt> sequenceStart = new ResetableReferenceState<>(mutInt -> new MutableInt(mutInt.intValue()), new MutableInt(0));
        final DataSetIterator sourceIter = new MockMiniEpochDataSetIterator() {

            private final int size = 5;
            @Override
            public DataSet next() {
                final int start = sequenceStart.get().intValue();
                sequenceStart.get().setValue(start + 13);
                return new DataSet(Nd4j.create(IntStream.range(start, start+size).mapToDouble(i -> i).toArray()), null);
            }
        };
        final MiniEpochDataSetIterator testIter = new AsynchEnablingDataSetIterator(sourceIter, sequenceStart, 2);

        assertTrue("Expect to have next!", testIter.hasNext());
        final DataSet first = testIter.next();

        assertTrue("Expect to have next!", testIter.hasNext());
        final DataSet second = testIter.next();
        // Test case does not test anything if first and second are equal
        assertNotEquals("Incorrect data!", first.getFeatures(), second.getFeatures());

        assertFalse("Does not expect to have next!", testIter.hasNext());

        testIter.restartMiniEpoch();
        assertTrue("Expect to have next!", testIter.hasNext());
        final DataSet firstAgain = testIter.next();
        assertEquals("Incorrect data!", first.getFeatures(), firstAgain.getFeatures());

        assertTrue("Expect to have next!", testIter.hasNext());
        final DataSet secondAgain = testIter.next();
        assertEquals("Incorrect data!", second.getFeatures(), secondAgain.getFeatures());

        testIter.reset(); // Allow for new batches to be created

        assertTrue("Expect to have next!", testIter.hasNext());
        final DataSet third = testIter.next();
        assertNotEquals("Incorrect data!", first.getFeatures(), third.getFeatures());

        assertTrue("Expect to have next!", testIter.hasNext());
        final DataSet fourth = testIter.next();
        assertNotEquals("Incorrect data!", second.getFeatures(), fourth.getFeatures());

        testIter.restartMiniEpoch();
        assertTrue("Expect to have next!", testIter.hasNext());
        final DataSet thirdAgain = testIter.next();
        assertEquals("Incorrect data!", third.getFeatures(), thirdAgain.getFeatures());

        assertTrue("Expect to have next!", testIter.hasNext());
        final DataSet fourthAgain = testIter.next();
        assertEquals("Incorrect data!", fourth.getFeatures(), fourthAgain.getFeatures());

    }

    @Override
    public void resetSupported() {
        assertTrue("Reset shall be supported!", decorate(new MockMiniEpochDataSetIterator()).resetSupported());
    }

    @Override
    public void asyncSupported() {
        assertTrue("Asynch shall be supported!", decorate(new MockMiniEpochDataSetIterator()).asyncSupported());
    }


    @Override
    public void hasNext() {
        assertTrue("Shall have next!", decorate(new MockMiniEpochDataSetIterator()).hasNext());
    }
}