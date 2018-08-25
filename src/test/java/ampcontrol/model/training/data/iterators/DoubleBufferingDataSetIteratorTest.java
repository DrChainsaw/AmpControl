package ampcontrol.model.training.data.iterators;

import org.apache.commons.lang.mutable.MutableBoolean;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.*;

/**
 * Test cases for {@link DoubleBufferingDataSetIterator}
 *
 * @author Christian Skärby
 */
public class DoubleBufferingDataSetIteratorTest extends DecoratingDataSetIteratorTest {


    @Override
    protected DataSetIterator decorate(MiniEpochDataSetIterator anIterator) {
        return new DoubleBufferingDataSetIterator(anIterator, 5);
    }

    /**
     * Test that next and reset work as intended
     */
    @Test
    public void nextAndReset() {
        final List<DataSet> sourceSets = IntStream.range(0, 10)
                .mapToObj(i -> Nd4j.create(new double[]{i}))
                .map(arr -> new DataSet(arr, arr))
                .collect(Collectors.toList());
        final DoubleBufferingDataSetIterator iter = new DoubleBufferingDataSetIterator(new MockDataSetIterator() {

            private int cnt = -1;

            @Override
            public boolean hasNext() {
                return true;
            }

            @Override
            public synchronized DataSet next() {
                cnt++;
                return sourceSets.get(cnt % sourceSets.size());
            }
        }, sourceSets.size() / 2);

        final List<DataSet> actualSets = new ArrayList<>();
        IntStream.range(0, sourceSets.size()).forEach(i -> {
            actualSets.add(iter.next());
        });
        assertEquals("Incorrect number of sets!", sourceSets.size(), actualSets.size());
        assertEquals("Incorrect data sets!", new HashSet<>(sourceSets), new HashSet<>(actualSets));

//        iter.reset();
//
//        final List<DataSet> actualSetsAgain = new ArrayList<>();
//        IntStream.range(0, sourceSets.size()).forEach(i -> actualSetsAgain.add(iter.next()));
//        assertEquals("Incorrect number of sets!", sourceSets.size(), actualSetsAgain.size());
//        assertEquals("Incorrect data sets!", new HashSet<>(sourceSets), new HashSet<>(actualSetsAgain));

    }

    @Test
    public void initCache() {
        final MutableBoolean wasCalled = new MutableBoolean(false);
        final DoubleBufferingDataSetIterator iter = new DoubleBufferingDataSetIterator(new MockDataSetIterator() {
            @Override
            public boolean hasNext() {
                return true;
            }

            @Override
            public DataSet next() {
                wasCalled.setValue(true);
                return new DataSet();
            }
        }, 5);
        assertFalse("Shall not have been called!", wasCalled.booleanValue());
        iter.initCache();
        sleep(100);
        assertTrue("Shall have been called!", wasCalled.booleanValue());
    }

    @Override
    public void resetSupported() {
        assertTrue("Reset shall be supported!", decorate(new MockMiniEpochDataSetIterator()).resetSupported());
    }

    private static void sleep(long ms) {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException e) {
            // Hope for the best then...
        }
    }
}