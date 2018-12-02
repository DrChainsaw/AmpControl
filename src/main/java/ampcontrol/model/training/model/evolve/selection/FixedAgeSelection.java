package ampcontrol.model.training.model.evolve.selection;

import ampcontrol.model.training.model.CompGraphAdapter;
import ampcontrol.model.training.model.evolve.mutate.util.CompGraphUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Removes candidates which have exceeded a given max age. This regularizes evolutionary algorithms. Concept from
 * https://arxiv.org/abs/1802.01548.
 * @param <T>
 *
 * @author Christian Skärby
 */
public class FixedAgeSelection<T, V> implements Selection<T> {

    private static final Logger log = LoggerFactory.getLogger(FixedAgeSelection.class);

    private final Function<T, V> toHashable;
    private final Selection<T> sourceSelection;
    private final int maxAge;
    private final Map<V, Integer> ageMap;

    /**
     * Constructor
     * @param toHashable Function to turn a candidate into a hashable object. Allows for different individuals to still
     *                   be considered the same, e.g. if the an individual is mutated but mutation did not change anything
     * @param sourceSelection Source selection. Will not see any candidates older than maxAge
     * @param maxAge Candidates seen more than this number of times will be removed
     */
    public FixedAgeSelection(Function<T, V> toHashable, Selection<T> sourceSelection, int maxAge) {
        this(new HashMap<>(), toHashable, sourceSelection, maxAge);
    }

    /**
     * Constructor
     * @param ageMap Mapping between hashable object and candidate age
     * @param toHashable Function to turn a candidate into a hashable object. Allows for different individuals to still
     *                   be considered the same, e.g. if the an individual is mutated but mutation did not change anything
     * @param sourceSelection Source selection. Will not see any candidates older than maxAge
     * @param maxAge Candidates seen more than this number of times will be removed
     */
    public FixedAgeSelection(Map<V, Integer> ageMap, Function<T, V> toHashable, Selection<T> sourceSelection, int maxAge) {
        this.ageMap = ageMap;
        this.toHashable = toHashable;
        this.sourceSelection = sourceSelection;
        this.maxAge = maxAge;
    }

    @Override
    public Stream<T> selectCandiates(List<Map.Entry<Double, T>> fitnessCandidates) {
        final Map<T, V> hashables = fitnessCandidates.stream()
                .map(Map.Entry::getValue)
                .collect(Collectors.toMap(
                        t -> t,
                        toHashable
                ));

        ageMap.keySet().retainAll(hashables.values());

        return sourceSelection.selectCandiates(
                fitnessCandidates.stream()
                        .peek(entry -> updateAge(hashables.get(entry.getValue())))
                        .filter(entry -> checkAge(entry, hashables, fitnessCandidates))
                        .collect(Collectors.toList()));
    }

    private void updateAge(V candidate) {
        ageMap.computeIfPresent(candidate, (cand, age) -> age + 1);
        ageMap.putIfAbsent(candidate, 0);
    }

    private boolean checkAge(Map.Entry<Double, T> candidate, Map<T, V> hashables, List<Map.Entry<Double, T>> allCandidates) {
        final V hashed = hashables.get(candidate.getValue());
        if(ageMap.get(hashed) < maxAge) {
            return true;
        }
        log.info("Discard candidate " + allCandidates.indexOf(candidate) + " due to old age");
        return false;
    }

    /**
     * Creates a {@link FixedAgeSelection} where two individuals are considered the same for ageing purposes if they have the
     * exact same ComputationGraphConfiguration.
     * @param maxAge Maximum age
     * @param sourceSelection source selection
     * @return a new FixedAgeSelection
     */
    public static <T extends CompGraphAdapter> FixedAgeSelection<T, String> byConfig(
            int maxAge,
            Map<String, Integer> ageMap,
            Selection<T> sourceSelection) {
        return new FixedAgeSelection<>(
                ageMap,
                adapter -> CompGraphUtil.configUniquenessString(adapter.asModel()),
                sourceSelection,
                maxAge
        );
    }
}
