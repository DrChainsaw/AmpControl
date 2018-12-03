package ampcontrol.model.training.model.evolve.state;

import lombok.Getter;
import org.apache.commons.lang3.mutable.Mutable;
import org.apache.commons.lang3.mutable.MutableObject;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.fail;

public class GenericStateTest {

    @Test
    public void mutateClone() {
        final Mutable<String> state = new MutableObject<>("pass");
        final AccessibleState<Mutable<String>> mutationState = createSaveStateMutation(state, (s,d) -> {/* */}).clone();
        state.setValue("fail");
        assertEquals("Incorrect output!", "pass", mutationState.get().getValue());
    }


    @Test
    public void save() {
        final GenericStateTest.SaveProbe probe = new GenericStateTest.SaveProbe("pass");
        final PersistentState<Mutable<String>> mutationState = createSaveStateMutation(new MutableObject<>("pass"), probe);
        try {
            mutationState.save("test");
            assertEquals("Incorrect save str", "test", probe.getLastString());
        } catch (IOException e) {
            fail("Unexpected exception!");
        }
    }
    
    private static AccessibleState<Mutable<String>> createSaveStateMutation(Mutable<String> startState, GenericState.PersistState<Mutable<String>> saveOperation) {
        return new GenericState<>(
                startState,
                state -> new MutableObject<>(state.getValue()),
                saveOperation);
    }

    @Getter
    private static final class SaveProbe implements GenericState.PersistState<Mutable<String>> {

        private String lastString = "NONE";
        private final String expected;

        SaveProbe(String expected) {
            this.expected = expected;
        }

        @Override
        public void save(String s, Mutable<String> state) {
            lastString = s;
            Assert.assertEquals("Incorrect state to save!", expected, state.getValue());
        }
    }
}