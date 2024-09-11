(G-CODE GENERATED BY FLATCAM v8.994 - www.flatcam.org - Version Date: 2020/11/7)

(Name: Drill_PTH_Through.DRL_cnc)
(Type: G-code from Geometry)
(Units: MM)

(Created on Tuesday, 10 September 2024 at 16:30)

(This preprocessor is the default preprocessor used by FlatCAM.)
(It is made to work with MACH3 compatible motion controllers.)


(TOOLS DIAMETER: )
(Tool: 1 -> Dia: 1.30002)
(Tool: 2 -> Dia: 1.50002)

(FEEDRATE Z: )
(Tool: 1 -> Feedrate: 300)
(Tool: 2 -> Feedrate: 300)

(FEEDRATE RAPIDS: )
(Tool: 1 -> Feedrate Rapids: 1500)
(Tool: 2 -> Feedrate Rapids: 1500)

(Z_CUT: )
(Tool: 1 -> Z_Cut: -2.2)
(Tool: 2 -> Z_Cut: -2.2)

(Tools Offset: )
(Tool: 1 -> Offset Z: 0.0)
(Tool: 2 -> Offset Z: 0.0)

(Z_MOVE: )
(Tool: 1 -> Z_Move: 2)
(Tool: 2 -> Z_Move: 2)

(Z Toolchange: 15 mm)
(X,Y Toolchange: 0.0000, 0.0000 mm)
(Z Start: None mm)
(Z End: 0.5 mm)
(X,Y End: None mm)
(Steps per circle: 64)
(Preprocessor Excellon: default)

(X range:    2.9330 ...   23.3830  mm)
(Y range:   45.3240 ...   93.0391  mm)

(Spindle Speed: 0 RPM)
G21
G90
G94

G01 F300.00

M5
G00 Z15.0000
T1
G00 X0.0000 Y0.0000                
M6
(MSG, Change to Tool Dia = 1.3000 ||| Total drills for tool T1 = 13)
M0
G00 Z15.0000

G01 F300.00
M03
G00 X22.7330 Y45.9740
G01 Z-2.2000
G01 Z0
G00 Z2.0000
G00 X22.7330 Y48.5140
G01 Z-2.2000
G01 Z0
G00 Z2.0000
G00 X22.7330 Y51.0540
G01 Z-2.2000
G01 Z0
G00 Z2.0000
G00 X20.4470 Y87.2889
G01 Z-2.2000
G01 Z0
G00 Z2.0000
G00 X20.4470 Y92.2889
G01 Z-2.2000
G01 Z0
G00 Z2.0000
G00 X3.6830 Y92.2891
G01 Z-2.2000
G01 Z0
G00 Z2.0000
G00 X3.6830 Y87.2891
G01 Z-2.2000
G01 Z0
G00 Z2.0000
G00 X3.6830 Y82.0021
G01 Z-2.2000
G01 Z0
G00 Z2.0000
G00 X3.6830 Y77.0021
G01 Z-2.2000
G01 Z0
G00 Z2.0000
G00 X3.6830 Y62.1901
G01 Z-2.2000
G01 Z0
G00 Z2.0000
G00 X3.6830 Y57.1901
G01 Z-2.2000
G01 Z0
G00 Z2.0000
G00 X3.6830 Y51.9031
G01 Z-2.2000
G01 Z0
G00 Z2.0000
G00 X3.6830 Y46.9031
G01 Z-2.2000
G01 Z0
G00 Z2.0000
M05
G00 Z0.50


