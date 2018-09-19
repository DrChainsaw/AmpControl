package ampcontrol.model.training.model.evolve.mutate;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link AggMutation}
 *
 * @author Christian Skärby
 */
public class AggMutationTest {

    /**
     * Test that mutate happens in right order
     */
    @Test
    public void mutate() {
        final Mutation<String> mutation = AggMutation.<String>builder()
                .first(new AppendMutation("_first"))
                .second(new AppendMutation("_second"))
                .build();
        assertEquals("Incorrect output!", "begin_first_second", mutation.mutate("begin"));
    }

    /**
     * Test that the builder works as expected
     */
    @Test
    public void builder() {
        final Mutation<String> mutation = AggMutation.<String>builder()
                .andThen(new AppendMutation("_first"))
                .andThen(new AppendMutation("_second"))
                .andThen(new AppendMutation("_third"))
                .andThen(new AppendMutation("_fourth"))
                .andThen(new AppendMutation("_fifth"))
                .build();
        assertEquals("Inoorrect output!", "begin_first_second_third_fourth_fifth", mutation.mutate("begin"));
    }

    private static class AppendMutation implements Mutation<String> {

        private final String append;

        private AppendMutation(String append) {
            this.append = append;
        }

        @Override
        public String mutate(String toMutate) {
            return toMutate + append;
        }
    }
}