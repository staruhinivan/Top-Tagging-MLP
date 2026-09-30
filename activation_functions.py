import torch
import torch.nn as nn
import numpy as np

# ==== Function ====
#=== Trainable activations === 
class Sawtooth(nn.Module):
    """
    Кусочно-гладкая периодическая функция с наклоном ±1
    
    Форма: пилообразная волна (sawtooth wave)
    f(x) = x mod period, но с чередованием наклона +1 и -1
    
    """
    def __init__(self, per_init=4.0, trainable=False):
        super().__init__()
        if trainable:
            self.T_saw = nn.Parameter(torch.tensor(per_init, dtype=torch.float32))
        else:
            self.T_saw=per_init

    
    def forward(self, x):
        # T = torch.clamp(self.T_saw, min=0.01)
        # Нормируем к периоду
        x_norm = (x) / self.T_saw +0.25     
        # Пилообразная волна с чередованием наклона
        # Используем fractional part для создания пилы
        fractional = x_norm - torch.floor(x_norm)        
        # Чередование наклона: +1 на [0,0.5), -1 на [0.5,1)
        output = torch.where(
            fractional < 0.5,
            fractional,           # наклон +1, от 0 до 1
            (1 - fractional)      # наклон -1, от 1 до 0
        )        
        # Масштабируем и центрируем
        output = (2 * output - 1)        
        return output
    
class Saw(nn.Module):
    def __init__(self, per_init=4.0, trainable=False):
        super().__init__()
        if trainable:
            self.Saw_per = nn.Parameter(torch.tensor(per_init, dtype=torch.float32))
        else:
            self.Saw_per=per_init

    def forward(self, X):
        # T = torch.clamp(self.period, min=0.01)      
        x_norm = X/self.Saw_per
        fract = x_norm - torch.floor(x_norm)
        fract = 2*fract - 1
        return fract
    
class PolRand(nn.Module):
    def __init__(self, seed, num_points = 11):
        super().__init__()
        self.points = num_points
        self.seed = seed
    def forward(self, X):
        x = np.arange(-self.points/2.0, self.points/2.0, 1.0)*0.5
        np.random.seed(self.seed)
        y = np.random.uniform(-1, 1, self.points)
        coef = np.polyfit(x, y, self.points)

        coef = torch.tensor(coef)
        result = torch.zeros_like(X)
        x_pow = torch.ones_like(X)
        coef = coef.flip(0)

        for i in range(len(coef)):
            result = result + coef[i]*x_pow
            x_pow = x_pow * X
        # e =np.exp(-np.fabs(X)/4)
        # e = torch.tensor(e)
        result = result * torch.exp(-torch.abs(X))
        return result
    
class PolPeriodic(nn.Module):
    '''периодическая парабола'''
    def __init__(self, per_init=4.0, trainable=False):
        super().__init__()
        if trainable:
            self.T_parab = nn.Parameter(torch.tensor(per_init, dtype=torch.float32))
        else:
            self.T_parab=per_init
    def forward(self, X):
        # T = torch.clamp(self.T_parab, min=0.01)
        x_norm = X / self.T_parab
        fractional = x_norm - torch.floor(x_norm)   
        output = torch.where(
            fractional < 0.5,
            -fractional*(fractional-0.5),           # наклон +1, от 0 до 1
            (fractional-1)*(fractional-0.5)      # наклон -1, от 1 до 0
        )  
        return output*20
    
class ChebyshevPolynomial(nn.Module):
    """
    Полином Чебышёва 1-го рода T_n(x)
    
    Рекуррентная формула:
    T_0(x) = 1
    T_1(x) = x
    T_{n+1}(x) = 2x * T_n(x) - T_{n-1}(x)
    """
    def __init__(self, degree=5, scale=1.0, trainable=False):
        super().__init__()
        self.degree = degree
        
        # Предвычисленные коэффициенты для быстрого вычисления
        self.coeffs = self._compute_coefficients(degree)
        if trainable:
            self.Scale_Chebysh = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        else:
            self.Scale_Chebysh=scale
    
    def _compute_coefficients(self, degree):
        """
        Вычисляет коэффициенты полинома Чебышёва
        Возвращает список коэффициентов от младшей степени к старшей
        """
        if degree == 0:
            return [1.0]
        elif degree == 1:
            return [0.0, 1.0]
        
        # Используем рекуррентное соотношение
        prev = [0.0, 1.0]  # T_1
        curr = [-1.0, 0.0, 2.0]  # T_2 = 2x^2 - 1
        
        for n in range(3, degree + 1):
            # T_{n} = 2x * T_{n-1} - T_{n-2}
            next_poly = [0] * (n + 1)
            
            # 2x * T_{n-1} (сдвиг степени)
            for i, c in enumerate(curr):
                if i + 1 <= n:
                    next_poly[i + 1] += 2 * c
            
            # - T_{n-2}
            for i, c in enumerate(prev):
                next_poly[i] -= c
            
            prev, curr = curr, next_poly
        
        return curr
    
    def forward(self, x):
        """
        Вычисляет T_n(x) для каждого x
        """
        # S = torch.clamp(self.Scale_Chebysh, min=0.01)
        x=x*self.Scale_Chebysh
        # Используем рекуррентное соотношение для численной стабильности
        if self.degree == 0:
            return torch.ones_like(x)
        elif self.degree == 1:
            return x
        
        T_prev = torch.ones_like(x)  # T_0
        T_curr = x  # T_1
        
        for n in range(2, self.degree + 1):
            T_next = 2 * x * T_curr - T_prev
            T_prev, T_curr = T_curr, T_next
        exp = torch.exp(-torch.abs(x)**2)
        return T_curr