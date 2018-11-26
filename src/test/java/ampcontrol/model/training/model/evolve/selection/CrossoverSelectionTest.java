package ampcontrol.model.training.model.evolve.selection;

import ampcontrol.model.training.model.evolve.CrossBreeding;
import org.junit.Test;

import java.util.AbstractMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link CrossoverSelection}
 *
 * @author Christian Skärby
 */
public class CrossoverSelectionTest {

    /**
     * Test that candidates selected by the source {@link Selection} are cross-bred.
     */
    @Test
    public void selectCandiates() {

        final List<Map.Entry<Double, StringCrossBreeding>> candidates = Stream.of
                ("ignore", "the", "dummy", "candidates", ",", "they", "shall", "not", "be", "selected", "and", "are", "not", "crossed")
                .map(StringCrossBreeding::new)
                .map(str -> new AbstractMap.SimpleEntry<>(666d, str))
                .collect(Collectors.toList());

        final Map<String, String> pairing = new LinkedHashMap<>();
        pairing.put("the", "selected");
        pairing.put("they", "are");
        pairing.put("be", "crossed");

        final String expected = "the selected candidates , they are selected and are crossed";
        final String actual = new CrossoverSelection<StringCrossBreeding>(
                cand -> !cand.string.equals("selected")
                        && !cand.string.equals(",")
                        && !cand.string.equals("and")
                        && !cand.string.equals("are")
                        && !cand.string.equals("candidates")
                        && !cand.string.equals("crossed"),

                (cand, cands) -> new StringCrossBreeding(pairing.get(cand.string)),

                cands -> cands.stream().map(Map.Entry::getValue)
                        .filter(cand -> !cand.string.equals("ignore"))
                        .filter(cand -> !cand.string.equals("dummy"))
                        .filter(cand -> !cand.string.equals("not"))
                        .filter(cand -> !cand.string.equals("nor"))
                        .filter(cand -> !cand.string.equals("as"))
                        .filter(cand -> !cand.string.equals("shall"))
                        .filter(cand -> !cand.string.equals("be"))
        ).selectCandiates(candidates).map(selected -> selected.string)
                .collect(Collectors.joining(" "));

        assertEquals("Incorrect selection!", expected, actual);
    }

    private final static class StringCrossBreeding implements CrossBreeding<StringCrossBreeding> {

        private final String string;

        private StringCrossBreeding(String string) {
            this.string = string;
        }

        @Override
        public StringCrossBreeding cross(StringCrossBreeding mate) {
            return new StringCrossBreeding(string + " " + mate.string);
        }
    }
}
