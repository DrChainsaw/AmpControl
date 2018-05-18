package ampcontrol.amp.labelmapping;

import ampcontrol.amp.midi.program.ProgramChange;
import org.junit.Test;

import javax.sound.midi.ShortMessage;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link MidiProgramChangeLabelMapping}
 *
 * @author Christian Skärby
 */
public class MidiProgramChangeLabelMappingTest {

    /**
     * Tests that the correct command is mapped
     */
    @Test
    public void apply() {
        // Note: Bank is not used by ProgramChange
        final ProgramChange cmd0 = createProgramChange(3, 0);
        final ProgramChange cmd1 = createProgramChange(45, 0);
        final ProgramChange cmd2 = createProgramChange(37, 0);
        final int channel = 11;
        MidiProgramChangeLabelMapping mapping = new MidiProgramChangeLabelMapping(channel, cmd0, cmd1, cmd2);

        final ShortMessage msg0 = mapping.apply(0).get(0);
        assertEquals("Incorrect channel!", channel, msg0.getChannel());
        assertEquals("Incorrect program!", cmd0.program(), msg0.getData1());

        final ShortMessage msg1 = mapping.apply(1).get(0);
        assertEquals("Incorrect channel!", channel, msg1.getChannel());
        assertEquals("Incorrect program!", cmd1.program(), msg1.getData1());

        final ShortMessage msg2 = mapping.apply(2).get(0);
        assertEquals("Incorrect channel!", channel, msg2.getChannel());
        assertEquals("Incorrect program!", cmd2.program(), msg2.getData1());

    }

    private static ProgramChange createProgramChange(final int program, final int bank) {
        return new ProgramChange() {
            @Override
            public int program() {
                return program;
            }

            @Override
            public int bank() {
                return bank;
            }
        };
    }
}