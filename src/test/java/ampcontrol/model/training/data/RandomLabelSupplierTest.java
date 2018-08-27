package ampcontrol.model.training.data;

import org.apache.commons.lang.mutable.MutableInt;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link RandomLabelSupplier}
 *
 * @author Christian Skärby
 */
public class RandomLabelSupplierTest {

    /**
     * Test that random elements are returned
     */
    @Test
    public void get() {
        final List<Integer> testList = IntStream.range(0, 10).boxed().collect(Collectors.toList());
        final List<Integer> expected = Arrays.asList(2,5,7,9,0);
        final Supplier<Integer> supplier = new RandomLabelSupplier<>(testList, new Random() {
            private int cnt = 0;
            @Override
            public int nextInt(int bound) {
                return expected.get(cnt++);
            }
        });
        List<Integer> result = Stream.generate(supplier).limit(expected.size()).collect(Collectors.toList());
        assertEquals("Incorrect result!", expected, result);
    }

    /**
     * Test that reset of random produces the same sequence again
     */
    @Test
    public void restoreState() {
        final List<Integer> testList = IntStream.range(0, 10).boxed().collect(Collectors.toList());
        final MutableInt cnt = new MutableInt(-1);
        final Supplier<Integer> supplier = new RandomLabelSupplier<>(testList, new Random() {
            @Override
            public int nextInt(int bound) {
                cnt.increment();
                return testList.get(cnt.intValue() % testList.size());
            }
        });
        List<Integer> expected = Stream.generate(supplier).limit(testList.size()-3).collect(Collectors.toList());
        cnt.setValue(-1);
        List<Integer> result = Stream.generate(supplier).limit(testList.size()-3).collect(Collectors.toList());
        assertEquals("Incorrect result!", expected, result);
    }
}