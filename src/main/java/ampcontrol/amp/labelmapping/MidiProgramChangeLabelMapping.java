package ampcontrol.amp.labelmapping;

import ampcontrol.amp.midi.program.ProgramChange;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.ShortMessage;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

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
                        throw new IllegalArgumentException("Failed to create message for prog: " + cmd.program() + " bank: " + cmd.bank() + "\n", e);
                    }
                })
                .collect(Collectors.toList());

    }

    @Override
    public List<ShortMessage> apply(Integer labelInd) {
        return Collections.singletonList(messageMap.get(labelInd));
    }
}
