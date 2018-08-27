package ampcontrol.model.training.data;

import ampcontrol.model.training.data.state.ResetableStateFactory;
import ampcontrol.model.training.data.state.SimpleStateFactory;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link CyclingLabelSupplier}
 *
 * @author Christian Skärby
 */
public class CyclingLabelSupplierTest {

    /**
     * Test that labels are produced in the expected order
     */
    @Test
    public void get() {
        final List<String> expected = Arrays.asList("qegftrh", "qwryry", "jyukuil", "htjy");
        final Supplier<String> supplier = new CyclingLabelSupplier<>(expected, new SimpleStateFactory(123));

        final List<String> actual = Stream.generate(supplier).limit(expected.size()).collect(Collectors.toList());
        assertEquals("Incorrect output!", expected, actual);

        final List<String> actualNext = Stream.generate(supplier).limit(expected.size()).collect(Collectors.toList());
        assertEquals("Incorrect output!", expected, actualNext);
    }

    /**
     * Test that same sequence is produced again when calling reset
     */
    @Test
    public void restoreState() {
        final List<Integer> testList = IntStream.range(0,17).boxed().collect(Collectors.toList());
        final ResetableStateFactory state = new ResetableStateFactory(123);
        final Supplier<Integer> supplier = new CyclingLabelSupplier<>(testList, state);

        final int nrToDraw = testList.size() - 3;
        final List<Integer> first = Stream.generate(supplier).limit(nrToDraw).collect(Collectors.toList());
        state.restorePreviousState();
        final List<Integer> second = Stream.generate(supplier).limit(nrToDraw).collect(Collectors.toList());
        assertEquals("Incorrect output!", first, second);
    }
}