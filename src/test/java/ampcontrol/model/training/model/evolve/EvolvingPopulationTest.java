package ampcontrol.model.training.model.evolve;

import org.junit.Test;

import java.util.*;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;

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
        final ControlledCallback evoCallback = new ControlledCallback();
        final EvolvingPopulation<String> evolving = new EvolvingPopulation<>(population,
                evoCallback,
                list -> list.stream()
                        .map(Map.Entry::getValue)
                        .map(StringBuilder::new)
                .map(StringBuilder::reverse)
                .map(StringBuilder::toString));

        assertTrue("No items expected!", evoCallback.currentPopulation.isEmpty());

        evolving.initEvolvingPopulation();
        assertEquals("Wrong population!", initialPopulation, evoCallback.currentPopulation);

        evoCallback.go();
        assertEquals("Wrong population!", initialPopulation.stream().map(StringBuilder::new)
                .map(StringBuilder::reverse)
                .map(StringBuilder::toString)
        .collect(Collectors.toList()), population);

        evoCallback.go();
        assertEquals("Wrong population!", initialPopulation, population);
    }

    /**
     * Test that an exception is thrown if evolve is called before any fitness is reported
     */
    @Test(expected = IllegalStateException.class)
    public void evolveToEarly() {
        new EvolvingPopulation<>(new ArrayList<>(Collections.singletonList("dummy")),
                (list, callback) -> {/* Does not matter*/},
                list -> Stream.of("Does not matter"))
                .evolve();
    }

    /**
     * Test that an exception is thrown if an unknown candidate is added
     */
    @Test(expected = IllegalArgumentException.class)
    public void unknownCandidate() {
        final ControlledCallback evoCallback = new ControlledCallback();
        new EvolvingPopulation<>(new ArrayList<>(Collections.singletonList("Valid member")),
                evoCallback,
                list -> Stream.of("Does not matter"))
                .initEvolvingPopulation();
        evoCallback.callback.accept(666d, "Invalid member");
    }

    /**
     * Test that an exception is thrown if an unknown candidate is added
     */
    @Test(expected = IllegalArgumentException.class)
    public void duplicateCandidate() {
        final String item = "Valid member";
        final ControlledCallback evoCallback = new ControlledCallback();
        new EvolvingPopulation<>(new ArrayList<>(Collections.singletonList(item)),
                evoCallback,
                list -> Stream.of("Does not matter"))
                .initEvolvingPopulation();
        evoCallback.callback.accept(666d, item);
        evoCallback.callback.accept(333d, item);
    }

    private final class ControlledCallback implements EvolvingPopulation.EvolutionCallback<String> {

        private List<String> currentPopulation = new ArrayList<>();
        private BiConsumer<Double, String> callback = (d,s) -> {};

        @Override
        public void accept(List<String> currentPopulation, BiConsumer<Double, String> callback) {
            this.currentPopulation = new ArrayList<>(currentPopulation);
            this.callback = callback;
        }

        private void go() {
            currentPopulation.forEach(item -> callback.accept((double)item.length(), item));
        }
    }
}