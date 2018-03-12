package ampControl.amp.labelmapping;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.ShortMessage;

import ampControl.amp.midi.program.ProgramChange;

/**
 * {@link LabelMapping} which maps label indexes to {@link ShortMessage ShortMessages} for MIDI devices.
 *
 * @author Christian Skärby
 */
public class MidiProgramChangeLabelMapping implements LabelMapping<ShortMessage> {

    private final List<ShortMessage> messageMap;

    public MidiProgramChangeLabelMapping(final int channel, ProgramChange... cmds) {
        messageMap = Stream.of(cmds)
                .sequential()
                .map(cmd -> {
                    try {
                        return new ShortMessage(ShortMessage.PROGRAM_CHANGE, channel, cmd.program(), cmd.bank());
                    } catch (InvalidMidiDataException e) {
                        throw new RuntimeException("Failed to create message for prog: " + cmd.program() + " bank: " + cmd.bank() + "\n" + e);
                    }
                })
                .collect(Collectors.toList());

    }

    @Override
    public List<ShortMessage> apply(Integer labelInd) {

        return Arrays.asList(messageMap.get(labelInd));
    }
}
