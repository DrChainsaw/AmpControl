package ampcontrol.amp.midi.program;

/**
 * Convenience enum for PodXt program change commands.
 *
 * @author Christian Skärby
 */
public enum PodXtProgramChange implements ProgramChange {

 A1(0),B1(1),C1(2),D1(3),A2(4),B2(5),C2(6),D2(7),A3(8),B3(9),C3(10),D3(11),A4(12),B4(13),C4(14),D4(15),A5(16),B5(17),C5(18),D5(19),A6(20),B6(21),C6(22),D6(23),A7(24),B7(25),C7(26),D7(27),A8(28),B8(29),C8(30),A9(32),B9(33),C9(34),D9(35),A10(36),B10(37),C10(38),D10(39),A11(40),B11(41),C11(42),D11(43),A12(44),B12(45),C12(46),D12(47),A13(48),B13(49),C13(50),D13(51),A14(52),B14(53),C14(54),D14(55),A15(56),B15(57),C15(58),D15(59),A16(60),B16(61),C16(62),A17(64),B17(65),C17(66),D17(67),A18(68),B18(69),C18(70),D18(71),A19(72),B19(73),C19(74),D19(75),A20(76),B20(77),C20(78),D20(79),A21(80),B21(81),C21(82),D21(83),A22(84),B22(85),C22(86),D22(87),A23(88),B23(89),C23(90),D23(91),A24(92),B24(93),C24(94),A25(96),B25(97),C25(98),D25(99),A26(100),B26(101),C26(102),D26(103),A27(104),B27(105),C27(106),D27(107),A28(108),B28(109),C28(110),D28(111),A29(112),B29(113),C29(114),D29(115),A30(116),B30(117),C30(118),D30(119),A31(120),B31(121),C31(122),D31(123),A32(124),B32(125),C32(126);

    private final int command;
    PodXtProgramChange(int command) {
        this.command = command;
    }

    @Override
    public int program() {
        return command;
    }

    @Override
    public int bank() {
        return 0;
    }
}
