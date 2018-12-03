package ampcontrol.model.training.model.evolve.selection;

import ampcontrol.model.training.model.evolve.Evolving;

import java.util.AbstractMap;
import java.util.List;
import java.util.Map;
import java.util.function.IntToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Mock {@link Evolving} for testing purposes
 */
class MockEvolvingItem implements Evolving<MockEvolvingItem> {

    private final String name;

    MockEvolvingItem(String name) {
        this.name = name;
    }

    @Override
    public MockEvolvingItem evolve() {
        return new MockEvolvingItem(evolvedName());
    }

    @Override
    public String toString() {
        return name;
    }

    String evolvedName() {
        return name + "_" + name;
    }

    static List<Map.Entry<Double, MockEvolvingItem>> createFitnessCands(int nrToCreate, IntToDoubleFunction fitnessMap) {
        return IntStream.range(0, nrToCreate)
                .mapToObj(i -> new AbstractMap.SimpleImmutableEntry<>(fitnessMap.applyAsDouble(i), new MockEvolvingItem("" + (i))))
                .collect(Collectors.toList());
    }
}
