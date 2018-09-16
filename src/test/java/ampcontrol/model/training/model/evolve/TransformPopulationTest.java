package ampcontrol.model.training.model.evolve;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.function.Function;
import java.util.stream.Collectors;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link TransformPopulation}
 *
 * @author Christian Skärby
 */
public class TransformPopulationTest {

    /**
     * Test conversion from {@link String} to {@link Integer}.
     */
    @Test
    public void streamPopulation() {
        final Population<Integer> population =
                new TransformPopulation<>(String::length,
                        new ListPopulation<>(Arrays.asList("", "a", "aa", "aaa")));
        assertEquals("Incorrect output!", Arrays.asList(0, 1, 2, 3), population.streamPopulation().collect(Collectors.toList()));
    }

    /**
     * Test that onChangeCallback is forwared to source
     */
    @Test
    public void onChangeCallback() {
        final ListPopulation<String> source = new ListPopulation<>(Collections.singletonList(""));
        final int[] nrofCallbacks = {0};
        new TransformPopulation<>(Function.identity(),
                source)
        .onChangeCallback(() -> nrofCallbacks[0]++);

        assertEquals("Incorrect number of callbacks!", 1 , source.getCallbacks().size());
        source.getCallbacks().forEach(Runnable::run);
        assertEquals("Incorrect number of callbacks!", 1, nrofCallbacks[0]);
    }
}