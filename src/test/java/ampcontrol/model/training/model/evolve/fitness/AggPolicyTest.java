package ampcontrol.model.training.model.evolve.fitness;

import org.junit.Test;

import java.util.function.Consumer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Test cases for {@link AggPolicy}
 *
 * @author Christian Skärby
 */
public class AggPolicyTest {


    /**
     * Test that fitness policies are applied in the right order
     */
    @Test
    public void apply() {
        final FitnessPolicy<String> policy = AggPolicy.<String>builder()
                .first(new AppendToStringPolicy("_first"))
                .second(new AppendToStringPolicy("_second"))
                .build();
        assertEquals("Incorrect output!", "begin_first_second", policy.apply("begin",
                dummy -> fail("not expected!")));
    }


    /**
     * Test that the builder works as expected
     */
    @Test
    public void builder() {
        final FitnessPolicy<String> policy = AggPolicy.<String>builder()
                .andThen(new AppendToStringPolicy("_first"))
                .andThen(new AppendToStringPolicy("_second"))
                .andThen(new AppendToStringPolicy("_third"))
                .andThen(new AppendToStringPolicy("_fourth"))
                .andThen(new AppendToStringPolicy("_fifth"))
                .build();
        assertEquals("Inoorrect output!", "begin_first_second_third_fourth_fifth",  policy.apply("begin",
                dummy -> fail("not expected!")));
    }

    private final class AppendToStringPolicy implements FitnessPolicy<String> {

        private final String toAppend;

        private AppendToStringPolicy(String toAppend) {
            this.toAppend = toAppend;
        }

        @Override
        public String apply(String candidate, Consumer<Double> fitnessListener) {
            return candidate + toAppend;
        }
    }
}