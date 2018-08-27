package ampcontrol.model.training.data.iterators.factory;

import ampcontrol.model.training.data.iterators.AsynchEnablingDataSetIterator;
import ampcontrol.model.training.data.iterators.MockDataSetIterator;
import ampcontrol.model.training.data.iterators.WorkSpaceWrappingIterator;
import org.junit.Test;

import java.util.stream.IntStream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link AutoFromSize}
 *
 * @author Christian Skärby
 */
public class AutoFromSizeTest {

    /**
     * Test that the right type of iterator is created. Probably not the "right" way to test a factory (is there one?)
     * Maybe use mock factories for set fits in memory vs set does not fit in memory?
     */
    @Test
    public void create() {
        final AutoFromSize.Input<Void> input = AutoFromSize.Input.<Void>builder()
                .sourceFactory(viod -> new MockDataSetIterator())
                .dataSetShape(new int[] {1,2,3})
                .batchSize(5)
                .dataSetSize(7)
                .build();

        final int expectedNrofCache = 11;
        final AutoFromSize<Void> factory = new AutoFromSize<>(2*3*5*7*(expectedNrofCache) * 11, 10);

        IntStream.range(0, expectedNrofCache).forEach(i ->
                assertEquals("Incorrect type returned after " + i + " calls!", WorkSpaceWrappingIterator.class, factory.create(input).getClass()));


        assertEquals("Incorrect type returned!", AsynchEnablingDataSetIterator.class, factory.create(input).getClass());

    }
}