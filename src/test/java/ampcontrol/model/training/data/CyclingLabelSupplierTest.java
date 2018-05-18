package ampcontrol.model.training.data;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;
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
        final Supplier<String> supplier = new CyclingLabelSupplier<>(expected);

        final List<String> actual = Stream.generate(supplier).limit(expected.size()).collect(Collectors.toList());
        assertEquals("Incorrect output!", expected, actual);

        final List<String> actualNext = Stream.generate(supplier).limit(expected.size()).collect(Collectors.toList());
        assertEquals("Incorrect output!", expected, actualNext);
    }
}