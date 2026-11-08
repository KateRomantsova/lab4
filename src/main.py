"""Запуск усієї лабораторної: python src/main.py"""
import part_a, part_b, part_c, part_d, part_e

if __name__ == "__main__":
    for p in (part_a, part_b, part_c, part_d, part_e):
        print("=" * 20, p.__name__)
        p.run()
    part_e.run_keyboard()
    part_e.run_stability()