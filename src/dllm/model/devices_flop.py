# sorts hardcoded and calculates flops if possible

class Flops:
    def __init__(self, fp32: float, fp16: float, int8: float):
        self.fp32: float = fp32
        self.fp16: float = fp16
        self.int8: float = int8

class DeviceFlops:
    def __init__(self, device_name: str):
        self.tflops = 1.0
        self.device_name = device_name
        
        self.apple_unified_chips = {
            "Apple M1": Flops(fp32=2.29*self.tflops, fp16=4.58*self.tflops, int8=9.16*self.tflops),
            "Apple M1 Pro": Flops(fp32=5.30*self.tflops, fp16=10.60*self.tflops, int8=21.20*self.tflops),
            "Apple M1 Max": Flops(fp32=10.60*self.tflops, fp16=21.20*self.tflops, int8=42.40*self.tflops),
            "Apple M1 Ultra": Flops(fp32=21.20*self.tflops, fp16=42.40*self.tflops, int8=84.80*self.tflops),
            "Apple M2": Flops(fp32=3.55*self.tflops, fp16=7.10*self.tflops, int8=14.20*self.tflops),
            "Apple M2 Pro": Flops(fp32=5.68*self.tflops, fp16=11.36*self.tflops, int8=22.72*self.tflops),
            "Apple M2 Max": Flops(fp32=13.49*self.tflops, fp16=26.98*self.tflops, int8=53.96*self.tflops),
            "Apple M2 Ultra": Flops(fp32=26.98*self.tflops, fp16=53.96*self.tflops, int8=107.92*self.tflops),
            "Apple M3": Flops(fp32=3.55*self.tflops, fp16=7.10*self.tflops, int8=14.20*self.tflops),
            "Apple M3 Pro": Flops(fp32=4.97*self.tflops, fp16=9.94*self.tflops, int8=19.88*self.tflops),
            "Apple M3 Max": Flops(fp32=14.20*self.tflops, fp16=28.40*self.tflops, int8=56.80*self.tflops),
            "Apple M3 Ultra": Flops(fp32=54.26*self.tflops, fp16=108.52*self.tflops, int8=217.04*self.tflops),
            "Apple M4": Flops(fp32=4.26*self.tflops, fp16=8.52*self.tflops, int8=17.04*self.tflops),
            "Apple M4 Pro": Flops(fp32=5.72*self.tflops, fp16=11.44*self.tflops, int8=22.88*self.tflops),
            "Apple M4 Max": Flops(fp32=18.03*self.tflops, fp16=36.07*self.tflops, int8=72.14*self.tflops),
            ### A chips
            "Apple A13 Bionic": Flops(fp32=0.69*self.tflops, fp16=1.38*self.tflops, int8=2.76*self.tflops),
            "Apple A14 Bionic": Flops(fp32=0.75*self.tflops, fp16=1.50*self.tflops, int8=3.00*self.tflops),
            "Apple A15 Bionic": Flops(fp32=1.37*self.tflops, fp16=2.74*self.tflops, int8=5.48*self.tflops),
            "Apple A16 Bionic": Flops(fp32=1.79*self.tflops, fp16=3.58*self.tflops, int8=7.16*self.tflops),
            "Apple A17 Pro": Flops(fp32=2.15*self.tflops, fp16=4.30*self.tflops, int8=8.60*self.tflops),
        }

        self.chip_flops = {
            "GPU": {
                ### NVIDIA GPUs
                "NVIDIA GEFORCE RTX 4090": Flops(fp32=82.58*self.tflops, fp16=165.16*self.tflops, int8=330.32*self.tflops),
                "NVIDIA GEFORCE RTX 4090 D": Flops(fp32=82.58*self.tflops, fp16=165.16*self.tflops, int8=330.32*self.tflops),
                "NVIDIA GEFORCE RTX 4080": Flops(fp32=48.74*self.tflops, fp16=97.48*self.tflops, int8=194.96*self.tflops),
                "NVIDIA GEFORCE RTX 4080 SUPER": Flops(fp32=52.0*self.tflops, fp16=104.0*self.tflops, int8=208.0*self.tflops),
                "NVIDIA GEFORCE RTX 4070 TI SUPER": Flops(fp32=40.0*self.tflops, fp16=80.0*self.tflops, int8=160.0*self.tflops),
                "NVIDIA GEFORCE RTX 4070 TI": Flops(fp32=39.43*self.tflops, fp16=78.86*self.tflops, int8=157.72*self.tflops),
                "NVIDIA GEFORCE RTX 4070 SUPER": Flops(fp32=30.0*self.tflops, fp16=60.0*self.tflops, int8=120.0*self.tflops),
                "NVIDIA GEFORCE RTX 4070": Flops(fp32=29.0*self.tflops, fp16=58.0*self.tflops, int8=116.0*self.tflops),
                "NVIDIA GEFORCE RTX 4060 TI 16GB": Flops(fp32=22.0*self.tflops, fp16=44.0*self.tflops, int8=88.0*self.tflops),
                "NVIDIA GEFORCE RTX 4060 TI": Flops(fp32=22.0*self.tflops, fp16=44.0*self.tflops, int8=88.0*self.tflops),
                "NVIDIA GEFORCE RTX 3050": Flops(fp32=9.11*self.tflops, fp16=18.22*self.tflops, int8=36.44*self.tflops),
                "NVIDIA GEFORCE RTX 3060": Flops(fp32=13.0*self.tflops, fp16=26.0*self.tflops, int8=52.0*self.tflops),
                "NVIDIA GEFORCE RTX 3060 TI": Flops(fp32=16.2*self.tflops, fp16=32.4*self.tflops, int8=64.8*self.tflops),
                "NVIDIA GEFORCE RTX 3070": Flops(fp32=20.3*self.tflops, fp16=40.6*self.tflops, int8=81.2*self.tflops),
                "NVIDIA GEFORCE RTX 3070 TI": Flops(fp32=21.8*self.tflops, fp16=43.6*self.tflops, int8=87.2*self.tflops),
                "NVIDIA GEFORCE RTX 3080 (10 GB)": Flops(fp32=29.8*self.tflops, fp16=59.6*self.tflops, int8=119.2*self.tflops),
                "NVIDIA GEFORCE RTX 3080 (12 GB)": Flops(fp32=30.6*self.tflops, fp16=61.2*self.tflops, int8=122.4*self.tflops),
                "NVIDIA GEFORCE RTX 3080 TI": Flops(fp32=34.1*self.tflops, fp16=68.2*self.tflops, int8=136.4*self.tflops),
                "NVIDIA GEFORCE RTX 3090": Flops(fp32=35.6*self.tflops, fp16=71.2*self.tflops, int8=142.4*self.tflops),
                "NVIDIA GEFORCE RTX 3090 TI": Flops(fp32=40.0*self.tflops, fp16=80.0*self.tflops, int8=160.0*self.tflops),
                "NVIDIA GEFORCE RTX 2060": Flops(fp32=6.45*self.tflops, fp16=12.9*self.tflops, int8=25.8*self.tflops),
                "NVIDIA GEFORCE RTX 2060 SUPER": Flops(fp32=7.2*self.tflops, fp16=14.4*self.tflops, int8=28.8*self.tflops),
                "NVIDIA GEFORCE RTX 2070": Flops(fp32=7.46*self.tflops, fp16=14.93*self.tflops, int8=29.86*self.tflops),
                "NVIDIA GEFORCE RTX 2070 SUPER": Flops(fp32=9.06*self.tflops, fp16=18.12*self.tflops, int8=36.24*self.tflops),
                "NVIDIA GEFORCE RTX 2080": Flops(fp32=10.07*self.tflops, fp16=20.14*self.tflops, int8=40.28*self.tflops),
                "NVIDIA GEFORCE RTX 2080 TI": Flops(fp32=13.45*self.tflops, fp16=26.9*self.tflops, int8=40.28*self.tflops),
                "NVIDIA GEFORCE RTX 2080 SUPER": Flops(fp32=11.15*self.tflops, fp16=22.30*self.tflops, int8=44.60*self.tflops),
                "NVIDIA TITAN RTX": Flops(fp32=16.31*self.tflops, fp16=32.62*self.tflops, int8=65.24*self.tflops),
                "NVIDIA GEFORCE GTX 1050 TI": Flops(fp32=2.0*self.tflops, fp16=4.0*self.tflops, int8=8.0*self.tflops),
                "NVIDIA GEFORCE GTX 1070": Flops(fp32=6.463*self.tflops, fp16=0.101*self.tflops, int8=25.852*self.tflops),
                "NVIDIA GEFORCE GTX 1080": Flops(fp32=8.873*self.tflops, fp16=0.138*self.tflops, int8=35.492*self.tflops),
                "NVIDIA GEFORCE GTX 1080 TI": Flops(fp32=11.34*self.tflops, fp16=0.177*self.tflops, int8=45.36*self.tflops),
                "NVIDIA GeForce GTX 1660 TI": Flops(fp32=4.8*self.tflops, fp16=9.6*self.tflops, int8=19.2*self.tflops),
                "NVIDIA RTX A2000": Flops(fp32=7.99*self.tflops, fp16=7.99*self.tflops, int8=31.91*self.tflops),
                "NVIDIA RTX A4000": Flops(fp32=19.17*self.tflops, fp16=19.17*self.tflops, int8=76.68*self.tflops),
                "NVIDIA RTX A4500": Flops(fp32=23.65*self.tflops, fp16=23.65*self.tflops, int8=94.6*self.tflops),
                "NVIDIA RTX A5000": Flops(fp32=27.8*self.tflops, fp16=27.8*self.tflops, int8=111.2*self.tflops),
                "NVIDIA RTX A6000": Flops(fp32=38.71*self.tflops, fp16=38.71*self.tflops, int8=154.84*self.tflops),
                "NVIDIA RTX 4000 ADA GENERATION": Flops(fp32=26.7*self.tflops, fp16=26.7*self.tflops, int8=258.0*self.tflops),
                "NVIDIA A40 48GB PCIE": Flops(fp32=37.4*self.tflops, fp16=149.7*self.tflops, int8=299.3*self.tflops),
                "NVIDIA A100 40GB PCIE": Flops(fp32=19.5*self.tflops, fp16=312.0*self.tflops, int8=624.0*self.tflops),
                "NVIDIA A800 40GB PCIE": Flops(fp32=19.5*self.tflops, fp16=312.0*self.tflops, int8=624.0*self.tflops),
                "NVIDIA A100 80GB PCIE": Flops(fp32=19.5*self.tflops, fp16=312.0*self.tflops, int8=624.0*self.tflops),
                "NVIDIA A800 80GB PCIE": Flops(fp32=19.5*self.tflops, fp16=312.0*self.tflops, int8=624.0*self.tflops),
                "NVIDIA A100 80GB SXM": Flops(fp32=19.5*self.tflops, fp16=312.0*self.tflops, int8=624.0*self.tflops),
                "NVIDIA A800 80GB SXM": Flops(fp32=19.5*self.tflops, fp16=312.0*self.tflops, int8=624.0*self.tflops),
                "NVIDIA T1000 8GB": Flops(fp32=2.5 * self.tflops, fp16=5.0 * self.tflops, int8=10.0 * self.tflops),
                "QUADRO M2000": Flops(fp32=0.5 * self.tflops, fp16=1.0 * self.tflops, int8=2.0 * self.tflops),
                "QUADRO P400": Flops(fp32=0.641 * self.tflops, fp16=1.282 * self.tflops, int8=2.564 * self.tflops),
                "NVIDIA A10": Flops(fp32=31.2 * self.tflops, fp16=62.5 * self.tflops, int8=2.5 * self.tflops),
                "JETSON AGX ORIN 32GB": Flops(fp32=17.65*self.tflops, fp16=35.3*self.tflops, int8=70.6*self.tflops),
                "JETSON AGX ORIN 64GB": Flops(fp32=24.27*self.tflops, fp16=48.54*self.tflops, int8=97.09*self.tflops),
                ### AMD GPUs
                "AMD Radeon RX 6900 XT": Flops(fp32=23.04*self.tflops, fp16=46.08*self.tflops, int8=92.16*self.tflops),
                "AMD Radeon RX 6800 XT": Flops(fp32=20.74*self.tflops, fp16=41.48*self.tflops, int8=82.96*self.tflops),
                "AMD Radeon RX 6800": Flops(fp32=16.17*self.tflops, fp16=32.34*self.tflops, int8=64.68*self.tflops),
                "AMD Radeon RX 6700 XT": Flops(fp32=13.21*self.tflops, fp16=26.42*self.tflops, int8=52.84*self.tflops),
                "AMD Radeon RX 6700": Flops(fp32=11.4*self.tflops, fp16=22.8*self.tflops, int8=45.6*self.tflops),
                "AMD Radeon RX 6600 XT": Flops(fp32=10.6*self.tflops, fp16=21.2*self.tflops, int8=42.4*self.tflops),
                "AMD Radeon RX 6600": Flops(fp32=8.93*self.tflops, fp16=17.86*self.tflops, int8=35.72*self.tflops),
                "AMD Radeon RX 6500 XT": Flops(fp32=5.77*self.tflops, fp16=11.54*self.tflops, int8=23.08*self.tflops),
                "AMD Radeon RX 6400": Flops(fp32=3.57*self.tflops, fp16=7.14*self.tflops, int8=14.28*self.tflops),
                "AMD Radeon RX 7900 XTX": Flops(fp32=61.4*self.tflops, fp16=122.8*self.tflops, int8=245.6*self.tflops),
                "AMD Radeon RX 7900 XT": Flops(fp32=53.4*self.tflops, fp16=106.8*self.tflops, int8=213.6*self.tflops),
                "AMD Radeon RX 7800 XT": Flops(fp32=42.6*self.tflops, fp16=85.2*self.tflops, int8=170.4*self.tflops),
                "AMD Radeon RX 7700 XT": Flops(fp32=34.2*self.tflops, fp16=68.4*self.tflops, int8=136.8*self.tflops),
                "AMD Radeon RX 7600": Flops(fp32=21.5*self.tflops, fp16=43.0*self.tflops, int8=86.0*self.tflops),
                "AMD Radeon RX 7500": Flops(fp32=16.2*self.tflops, fp16=32.4*self.tflops, int8=64.8*self.tflops),
                

                # Apple Silicons
                **self.apple_unified_chips
            },
            "CPU": {
                "Ryzen 7 2700": Flops(fp32=0.474*self.tflops, fp16=0.474*self.tflops, int8=0.947*self.tflops),
                "Intel Core i7-11800H": Flops(fp32=1.178*self.tflops, fp16=1.178*self.tflops, int8=4.710*self.tflops),
                "Intel Xeon W-2102": Flops(fp32=0.371*self.tflops, fp16=0.371*self.tflops, int8=0.742*self.tflops),
                **self.apple_unified_chips
            }   
        }

        def __str__(self):
            return f"Device: {self.device_name}, \
                fp32: {self.fp32 / self.tflops:.2f} TFLOPS, \
                fp16: {self.fp16 / self.tflops:.2f} TFLOPS, \
                int8: {self.int8 / self.tflops:.2f} TFLOPS"

    def get_flops(self) -> int:
        return self.flops

    