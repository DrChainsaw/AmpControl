package ampcontrol.amp.midi.program;

/**
 * Represents a program change in terms of a program and a bank.
 *
 * @author Christian Skärby
 */
public interface ProgramChange {
    /**
     * Program to change to. Must fit in a byt (range 0-127) to comply with the MIDI standard.
     * @return
     */
    int program(); // Why not byte? Because javas ShortMessage only accepts ints

    /**
     * Bank to use. Must fit in a byt (range 0-127) to comply with the MIDI standard.
     *
     * @return
     */
    int bank(); // Why not byte? Because javas ShortMessage only accepts ints
}
