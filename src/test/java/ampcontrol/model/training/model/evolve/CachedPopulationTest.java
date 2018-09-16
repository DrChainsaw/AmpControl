package ampcontrol.model.training.model.evolve;

import junit.framework.TestCase;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * Test cases for {@link CachedPopulation}
 *
 * @author Christian Skärby
 */
public class CachedPopulationTest {

    /**
     * Test that population is cached and that cache can be refreshed.
     */
    @Test
    public void streamPopulation() {
        final List<String> pop = new ArrayList<>(Arrays.asList("the", "cache", "is", "fresh"));
        final ListPopulation<String> source = new ListPopulation<>(pop);
        final CachedPopulation<String> cache = new CachedPopulation<>(source);

        assertEquals("Expected cache and source to be same!",
                source.streamPopulation().collect(Collectors.joining(" ")),
                cache.streamPopulation().collect(Collectors.joining(" ")));

        pop.remove(3);
        pop.add("refreshed");

        assertNotEquals("Expected cache and source to differ!",
                source.streamPopulation().collect(Collectors.joining(" ")),
                cache.streamPopulation().collect(Collectors.joining(" ")));

        source.getCallbacks().forEach(Runnable::run);

        assertEquals("Expected cache and source to be same!",
                source.streamPopulation().collect(Collectors.joining(" ")),
                cache.streamPopulation().collect(Collectors.joining(" ")));

    }

    /**
     * Test that callback works as expected
     */
    @Test
    public void onChangeCallback() {
        final ListPopulation<String> source = new ListPopulation<>(Collections.singletonList(""));
        final int[] nrofCallbacks = {0};
        final Population<String> cache = new CachedPopulation<>(source);
        cache.onChangeCallback(() -> nrofCallbacks[0]++);

        TestCase.assertEquals("Incorrect number of callbacks!", 1, source.getCallbacks().size());
        source.getCallbacks().forEach(Runnable::run);
        TestCase.assertEquals("Incorrect number of callbacks!", 0, nrofCallbacks[0]);
        assertEquals("Incorrect number of elements!", 1 , cache.streamPopulation().count());
        TestCase.assertEquals("Incorrect number of callbacks!", 1, nrofCallbacks[0]);
    }
}