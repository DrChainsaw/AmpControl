package ampcontrol.model.training.model.evolve;

import ampcontrol.model.training.model.evolve.fitness.FitnessPolicy;
import org.junit.Test;

import java.util.*;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * Test cases for {@link EvolvingPopulation}
 *
 * @author Christian Skärby
 */
public class EvolvingPopulationTest {

    /**
     * Test that population is evolved as expected.
     */
    @Test
    public void evolve() {
        final List<String> initialPopulation = Arrays.asList("This", "list", "will", "evolve", "in", "reverse");
        final List<String> population = new ArrayList<>(initialPopulation);
        final ControlledFitnessCallback fitnessCallback = new ControlledFitnessCallback();
        final EvolvingPopulation<String> evolving = new EvolvingPopulation<>(population,
                fitnessCallback,
                list -> list.stream()
                        .map(Map.Entry::getValue)
                        .map(StringBuilder::new)
                        .map(StringBuilder::reverse)
                        .map(StringBuilder::toString));

        assertEquals("Wrong population!", initialPopulation, evolving.streamPopulation().collect(Collectors.toList()));

        fitnessCallback.go();
        fitnessCallback.currentPopulation.clear();
        assertEquals("Wrong population!", initialPopulation.stream().map(StringBuilder::new)
                .map(StringBuilder::reverse)
                .map(StringBuilder::toString)
                .collect(Collectors.toList()), evolving.streamPopulation().collect(Collectors.toList()));
        
        fitnessCallback.go();
        assertEquals("Wrong population!", initialPopulation, evolving.streamPopulation().collect(Collectors.toList()));
    }

    /**
     * Test that call back on change works as expected
     */
    @Test
    public void onChangeCallback() {
        final ControlledFitnessCallback fitnessCallback = new ControlledFitnessCallback();
        final EvolvingPopulation<String> evolving = new EvolvingPopulation<>(
                new ArrayList<>(Collections.singletonList("dummy")),
                fitnessCallback,
                list -> Stream.of("Does not matter"));
        final int[] nrofOnChangeCallbacks = {0};
        evolving.onChangeCallback(() -> nrofOnChangeCallbacks[0]++);

        fitnessCallback.go();
        fitnessCallback.currentPopulation.clear();
        assertEquals("Incorrect number of calls", 1 , nrofOnChangeCallbacks[0]);

        evolving.evolve();
        fitnessCallback.go();
        fitnessCallback.currentPopulation.clear();
        assertEquals("Incorrect number of calls", 2 , nrofOnChangeCallbacks[0]);
    }

    /**
     * Test that an exception is thrown if evolve is called before any fitness is reported
     */
    @Test(expected = IllegalStateException.class)
    public void evolveToEarly() {
        new EvolvingPopulation<>(
                new ArrayList<>(Collections.singletonList("dummy")),
                (str, callback) -> str, // Does not matter
                list -> Stream.of("Does not matter"))
                .evolve();
    }

    /**
     * Test that an exception is thrown if an unknown candidate is added
     */
    @Test(expected = IllegalArgumentException.class)
    public void unknownCandidate() {
        final ControlledFitnessCallback fitnessCallback = new ControlledFitnessCallback();
        final EvolvingPopulation<String> evolving = new EvolvingPopulation<>(
                new ArrayList<>(Collections.singletonList("Previous valid member")),
                fitnessCallback,
                list -> Stream.of("New valid member"));
        final Map<String, Consumer<Double>> oldPop = new LinkedHashMap<>(fitnessCallback.currentPopulation);
        fitnessCallback.go();
        evolving.streamPopulation().forEachOrdered(candidate ->
                assertFalse(candidate + " should no longer be part of population!", oldPop.containsKey(candidate)));
        oldPop.values().forEach(action -> action.accept(666d));
    }

    /**
     * Test that an exception is thrown if a duplicate candidate is added
     */
    @Test(expected = IllegalArgumentException.class)
    public void duplicateCandidate() {
        final String item = "Valid member";
        final ControlledFitnessCallback fitnessCallback = new ControlledFitnessCallback();
        new EvolvingPopulation<>(
                new ArrayList<>(Collections.singletonList(item)),
                fitnessCallback,
                list -> Stream.of("Does not matter"));
        fitnessCallback.go();
        fitnessCallback.go();
    }

    private final class ControlledFitnessCallback implements FitnessPolicy<String> {

        private Map<String, Consumer<Double>> currentPopulation = new LinkedHashMap<>();


        private void go() {
            currentPopulation.forEach((item, callback) -> callback.accept((double) item.length()));
        }

        @Override
        public String apply(String candidate, Consumer<Double> fitnessListener) {
            currentPopulation.put(candidate, fitnessListener);
            return candidate;
        }
    }
}