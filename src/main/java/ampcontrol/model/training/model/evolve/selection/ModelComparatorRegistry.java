package ampcontrol.model.training.model.evolve.selection;

import org.deeplearning4j.nn.api.Model;

import java.util.*;
import java.util.function.Function;

/**
 * Holds a mapping between {@link Model}s and a mapping for comparators. Intended use case is to be able to inject
 * comparators created for a certain model in case comparator is created far away from where it is consumed.
 * <br><br>
 * Exists as a kind of hack as I can't find a better way to propagate this, probably due to some code smell which
 * hopefully will be discovered and cleaned up sooner or later.
 *
 * @author Christian Skärby
 */
public class ModelComparatorRegistry {

    private final Map<Model, ComparatorMapping> registry = new WeakHashMap<>();

    private final static class ComparatorMapping implements Function<String, Optional<Function<Integer, Comparator<Integer>>>> {

        private final Map<String, DimensionMapping> mapping = new HashMap<>();

        @Override
        public Optional<Function<Integer, Comparator<Integer>>> apply(String layerName) {
            return Optional.ofNullable(mapping.get(layerName));
        }

        /**
         * Add comparator
         * @param layerName layer to map comparator to
         * @param dimension dimension to map comparator to
         * @param comparator the comparator which shall be used for given layerName and dimension
         */
        private void add(String layerName, int dimension, Comparator<Integer> comparator) {
            final DimensionMapping dimensionMapping = mapping.getOrDefault(layerName, new DimensionMapping());
            dimensionMapping.add(dimension, comparator);
            mapping.put(layerName, dimensionMapping);
        }
    }

    private final static class DimensionMapping implements  Function<Integer, Comparator<Integer>> {

        private final Map<Integer, Comparator<Integer>> mapping = new HashMap<>();

        @Override
        public Comparator<Integer> apply(Integer dimension) {
            return mapping.get(dimension);
        }

        /**
         * Add comparator
         * @param dimension dimension to map comparator to
         * @param comparator the comparator which shall be used for given layerName and dimension
         */
        private void add(int dimension, Comparator<Integer> comparator) {
            if(mapping.containsKey(dimension)) {
                throw new IllegalStateException("May not overwrite mapping for dimension " + dimension);
            }
            mapping.put(dimension, comparator);
        }
    }

    public Function<String, Optional<Function<Integer, Comparator<Integer>>>> get(Model model) {
        return registry.get(model);
    }

    /**
     * Add comparator
     * @param model Model to map comparator to
     * @param layerName layer to map comparator to
     * @param dimension dimension to map comparator to
     * @param comparator the comparator which shall be used for given layerName and dimension
     */
    public void add(Model model, String layerName, int dimension, Comparator<Integer> comparator) {
        final ComparatorMapping compMapping = registry.getOrDefault(model, new ComparatorMapping());
        compMapping.add(layerName, dimension, comparator);
        registry.put(model, compMapping);
    }

    public void clear(Model model) {
        registry.remove(model);
    }


}
