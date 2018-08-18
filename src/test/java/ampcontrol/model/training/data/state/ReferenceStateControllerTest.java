package ampcontrol.model.training.data.state;

import org.junit.Test;

import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link ResetableReferenceState}
 *
 * @author Christian Skärby
 */
public class ReferenceStateControllerTest {

    @Test
    public void testStateStorage() {
        final String initial = "initial";
        final ResetableReferenceState<MutableString> state = new ResetableReferenceState<>(
                str -> new MutableString().setVal(str.val()),
                new MutableString().setVal(initial));

        assertEquals("Incorrect state!", initial, state.get().val());

        final String next0 = "next0";
        state.get().setVal(next0);
        assertEquals("Incorrect state!", next0, state.get().val());

        state.restorePreviousState();
        assertEquals("Incorrect state!", initial, state.get().val());

        state.get().setVal(next0);
        state.storeCurrentState();

        IntStream.rangeClosed(1, 7).mapToObj(i -> "next" + i).forEach(val -> state.get().setVal(val));
        assertEquals("Incorrect state!", "next7", state.get().val());

        state.restorePreviousState();
        assertEquals("Incorrect state!", next0, state.get().val());
    }

    private final static class MutableString {

        private String val = "";

        private String val() {
            return val;
        }

        private MutableString setVal(String val) {
            this.val = val;
            return this;
        }

    }

}