package ampcontrol.model.training.data;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Expands a collection by padding it with a predefined number of copies of each element. 0 copies means the element
 * will be removed. If multiple instances of an element is present each instance will be copied.
 *
 * @author Christian Skärby
 */
public class MultiplyLabelExpander implements Function<Collection<String>, List<String>> {

    private final Map<String, Integer> labelExpansion = new LinkedHashMap<>();


    @Override
    public List<String> apply(Collection<String> labels) {
        return labels.stream()
                .flatMap(label -> Collections.nCopies(labelExpansion.getOrDefault(label, 1), label).stream())
                .collect(Collectors.toList());
    }

    /**
     * Add label to copy. If multiple instances of an element is present in the given collection each instance will be
     * copied.  0 copies means the element will be removed.
     * @param label The label to copy
     * @param expansion How many copies to insert for each instance of the given label
     * @return The {@link MultiplyLabelExpander} instance to make api fluent.
     */
    public MultiplyLabelExpander addExpansion(String label, int expansion) {
        labelExpansion.put(label, expansion);
        return this;
    }

    public static void main(String[] args) {
        MultiplyLabelExpander exp = new MultiplyLabelExpander()
                .addExpansion("aa", 0)
                .addExpansion("cc", 3);

        System.out.println(exp.apply(Arrays.asList("qq", "aa", "bb", "cc", "dd")));
    }
}
