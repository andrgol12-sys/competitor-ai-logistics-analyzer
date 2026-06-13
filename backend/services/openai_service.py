"""
Сервис для работы с ProxyAPI (OpenAI-совместимый API)
https://proxyapi.ru/docs/openai-text-generation
"""
import base64
import json
import re
import time
import logging
from typing import Optional

from openai import OpenAI

from backend.config import settings
from backend.models.schemas import CompetitorAnalysis, ImageAnalysis

# Логгер для сервиса
logger = logging.getLogger("competitor_monitor.openai")


class OpenAIService:
    """Сервис для анализа через ProxyAPI"""
    
    def __init__(self):
        logger.info("=" * 50)
        logger.info("Инициализация OpenAI сервиса")
        logger.info(f"  Base URL: {settings.proxy_api_base_url}")
        logger.info(f"  Модель текста: {settings.openai_model}")
        logger.info(f"  Модель vision: {settings.openai_vision_model}")
        logger.info(f"  API ключ: {'*' * 10}...{settings.proxy_api_key[-4:] if settings.proxy_api_key else 'НЕ ЗАДАН'}")
        
        self.client = OpenAI(
            api_key=settings.proxy_api_key,
            base_url=settings.proxy_api_base_url
        )
        self.model = settings.openai_model
        self.vision_model = settings.openai_vision_model
        
        logger.info("OpenAI сервис инициализирован успешно ✓")
        logger.info("=" * 50)
    
    def _parse_json_response(self, content: str) -> dict:
        """Извлечь JSON из ответа модели"""
        logger.debug(f"Парсинг JSON ответа, длина: {len(content)} символов")
        
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            content = json_match.group(1)
            logger.debug("JSON найден в markdown блоке")
        
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            content = json_match.group(0)
            logger.debug("JSON объект извлечён")
        
        try:
            result = json.loads(content)
            logger.debug(f"JSON успешно распарсен, ключей: {len(result)}")
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"Ошибка парсинга JSON: {e}")
            logger.debug(f"Проблемный контент: {content[:200]}...")
            return {}
    
    async def analyze_text(self, text: str) -> CompetitorAnalysis:
        """Анализ текста конкурента"""
        logger.info("=" * 50)
        logger.info("📝 АНАЛИЗ ТЕКСТА КОНКУРЕНТА")
        logger.info(f"  Длина текста: {len(text)} символов")
        logger.info(f"  Превью: {text[:100]}...")
        logger.info(f"  Модель: {self.model}")
        
        system_prompt = """Ты — эксперт по конкурентному анализу транспортных компаний, специализирующихся на рефрижераторных перевозках 20 тонн.

Проанализируй предоставленный текст конкурента и верни структурированный JSON-ответ.

Формат ответа (строго JSON):
{
    "strengths": [
        "сильная сторона 1",
        "сильная сторона 2",
        "сильная сторона 3"
    ],
    "weaknesses": [
        "слабая сторона 1",
        "слабая сторона 2",
        "слабая сторона 3"
    ],
    "unique_offers": [
        "уникальное предложение 1",
        "уникальное предложение 2",
        "уникальное предложение 3"
    ],
    "recommendations": [
        "практическая рекомендация 1",
        "практическая рекомендация 2",
        "практическая рекомендация 3"
    ],
    "summary": "Краткое резюме анализа с точки зрения рынка рефрижераторных перевозок"
}

При анализе учитывай:
- рефрижераторные перевозки;
- транспорт 20 тонн;
- температурные режимы;
- географию перевозок;
- наличие собственного автопарка;
- работу с федеральными сетями и производителями продуктов питания;
- скорость подачи транспорта;
- доверие к компании;
- понятность коммерческого предложения;
- наличие УТП;
- качество аргументов для B2B-клиентов.

Важно:
- Верни только JSON без пояснений и markdown
- Каждый массив должен содержать 4-6 конкретных пунктов
- Пиши на русском языке
- Будь конкретен и практичен
- Давай actionable рекомендации
- Не придумывай факты, которых нет на скриншоте или в переданном тексте
- Если информации недостаточно, прямо укажи: "На сайте не обнаружено..."
- Не утверждай, что компания занимается рефрижераторными перевозками, если это явно не видно
- Не называй отсутствие рефрижераторных перевозок слабой стороной, если компания не позиционируется как рефрижераторный перевозчик
- Отличай факт от предположения"""

        start_time = time.time()
        logger.info("  Отправка запроса к API...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Проанализируй текст конкурента:\n\n{text}"}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            elapsed = time.time() - start_time
            logger.info(f"  ✓ Ответ получен за {elapsed:.2f} сек")
            
            content = response.choices[0].message.content
            logger.info(f"  Длина ответа: {len(content)} символов")
            logger.debug(f"  Использовано токенов: {response.usage.total_tokens if response.usage else 'N/A'}")
            
            data = self._parse_json_response(content)
            
            result = CompetitorAnalysis(
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                unique_offers=data.get("unique_offers", []),
                recommendations=data.get("recommendations", []),
                summary=data.get("summary", "")
            )
            
            logger.info(f"  Результат: {len(result.strengths)} сильных, {len(result.weaknesses)} слабых сторон")
            logger.info("=" * 50)
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"  ✗ Ошибка API за {elapsed:.2f} сек: {e}")
            logger.error("=" * 50)
            raise
    
    async def analyze_image(self, image_base64: str, mime_type: str = "image/jpeg") -> ImageAnalysis:
        """Анализ изображения (баннер, сайт, упаковка)"""
        logger.info("=" * 50)
        logger.info("🖼️ АНАЛИЗ ИЗОБРАЖЕНИЯ")
        logger.info(f"  Размер base64: {len(image_base64)} символов")
        logger.info(f"  MIME тип: {mime_type}")
        logger.info(f"  Модель: {self.vision_model}")
        
        system_prompt = """Ты — эксперт по визуальному маркетингу, UX/UI и конкурентному анализу сайтов транспортных компаний.

Проанализируй изображение конкурента: скриншот сайта, баннер, блок услуг, карточку преимущества или рекламный материал транспортной компании.

Формат ответа (строго JSON):
{
    "description": "Детальное описание того, что изображено",
    "marketing_insights": [
        "маркетинговый инсайт 1",
        "маркетинговый инсайт 2",
        "маркетинговый инсайт 3"
    ],
    "visual_style_score": 7,
    "visual_style_analysis": "Анализ визуального стиля, доверия и удобства интерфейса",
    "recommendations": [
        "практическая рекомендация 1",
        "практическая рекомендация 2",
        "практическая рекомендация 3"
    ]
}

При анализе оценивай:
- насколько сайт вызывает доверие у B2B-клиента;
- понятность услуг рефрижераторных перевозок;
- видимость преимуществ компании;
- качество первого экрана;
- наличие понятного CTA;
- современность дизайна;
- читаемость текста;
- наличие доказательств: автопарк, география, опыт, клиенты, отзывы, документы;
- удобство для клиента, который хочет быстро оставить заявку.

Важно:
- Верни только JSON без пояснений и markdown
- visual_style_score от 0 до 10
- Каждый массив должен содержать 3-5 пунктов
- Пиши на русском языке
- Рекомендации должны быть применимы для сайта транспортной компании"""

        start_time = time.time()
        logger.info("  Отправка запроса к Vision API...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Проанализируй это изображение конкурента с точки зрения маркетинга, доверия, UX/UI и продаж транспортных услуг:"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            elapsed = time.time() - start_time
            logger.info(f"  ✓ Ответ получен за {elapsed:.2f} сек")
            
            content = response.choices[0].message.content
            logger.info(f"  Длина ответа: {len(content)} символов")
            
            data = self._parse_json_response(content)
            
            result = ImageAnalysis(
                description=data.get("description", ""),
                marketing_insights=data.get("marketing_insights", []),
                visual_style_score=data.get("visual_style_score", 5),
                visual_style_analysis=data.get("visual_style_analysis", ""),
                recommendations=data.get("recommendations", [])
            )
            
            logger.info(f"  Результат: оценка стиля {result.visual_style_score}/10")
            logger.info(f"  Инсайтов: {len(result.marketing_insights)}, рекомендаций: {len(result.recommendations)}")
            logger.info("=" * 50)
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"  ✗ Ошибка Vision API за {elapsed:.2f} сек: {e}")
            logger.error("=" * 50)
            raise
    
    async def analyze_parsed_content(
        self, 
        title: Optional[str], 
        h1: Optional[str], 
        paragraph: Optional[str]
    ) -> CompetitorAnalysis:
        """Анализ распарсенного контента сайта"""
        logger.info("📄 Анализ распарсенного контента")
        logger.info(f"  Title: {title[:50] if title else 'N/A'}...")
        logger.info(f"  H1: {h1[:50] if h1 else 'N/A'}...")
        logger.info(f"  Абзац: {paragraph[:50] if paragraph else 'N/A'}...")
        
        content_parts = []
        if title:
            content_parts.append(f"Заголовок страницы (title): {title}")
        if h1:
            content_parts.append(f"Главный заголовок (H1): {h1}")
        if paragraph:
            content_parts.append(f"Первый абзац: {paragraph}")
        
        combined_text = "\n\n".join(content_parts)
        
        if not combined_text.strip():
            logger.warning("  ⚠ Контент пустой, возвращаем пустой анализ")
            return CompetitorAnalysis(
                summary="Не удалось извлечь контент для анализа"
            )
        
        return await self.analyze_text(combined_text)
    
    async def analyze_website_screenshot(
        self,
        screenshot_base64: str,
        url: str,
        title: Optional[str] = None,
        h1: Optional[str] = None,
        first_paragraph: Optional[str] = None
    ) -> CompetitorAnalysis:
        """Комплексный анализ сайта конкурента по скриншоту"""
        logger.info("=" * 50)
        logger.info("🌐 КОМПЛЕКСНЫЙ АНАЛИЗ САЙТА")
        logger.info(f"  URL: {url}")
        logger.info(f"  Title: {title[:50] if title else 'N/A'}...")
        logger.info(f"  H1: {h1[:50] if h1 else 'N/A'}...")
        logger.info(f"  Размер скриншота: {len(screenshot_base64)} символов base64")
        logger.info(f"  Модель: {self.vision_model}")
        
        context_parts = [f"URL сайта: {url}"]
        if title:
            context_parts.append(f"Title страницы: {title}")
        if h1:
            context_parts.append(f"Главный заголовок (H1): {h1}")
        if first_paragraph:
            context_parts.append(f"Текст на странице: {first_paragraph[:300]}")
        
        context = "\n".join(context_parts)
        logger.debug(f"  Контекст:\n{context}")
        
        system_prompt = """Ты — эксперт по конкурентному анализу, UX/UI и маркетингу транспортных компаний.

Проанализируй скриншот сайта конкурента в сфере транспортной логистики.
Если на сайте явно есть информация о рефрижераторных перевозках, температурных режимах или 20-тонных фурах — отдельно оцени это.
Если такой информации нет, не придумывай её.

Формат ответа (строго JSON):
{
    "strengths": [
        "сильная сторона сайта 1",
        "сильная сторона сайта 2",
        "сильная сторона сайта 3"
    ],
    "weaknesses": [
        "слабая сторона сайта 1",
        "слабая сторона сайта 2",
        "слабая сторона сайта 3"
    ],
    "unique_offers": [
        "заметное УТП или преимущество 1",
        "заметное УТП или преимущество 2",
        "заметное УТП или преимущество 3"
    ],
    "recommendations": [
        "практическая рекомендация 1",
        "практическая рекомендация 2",
        "практическая рекомендация 3"
    ],
    "summary": "Комплексное резюме анализа сайта конкурента"
}

При анализе обращай внимание на:
- насколько быстро понятно, чем занимается компания;
- есть ли признаки специализации компании: тип перевозок, отрасль, температурный режим, тип транспорта;
- оцени полноту описания услуг только в рамках того, что реально заявлено на сайте;
- насколько сайт вызывает доверие;
- есть ли доказательства: автопарк, опыт, клиенты, отзывы, документы;
- насколько заметны CTA-кнопки;
- насколько удобно оставить заявку;
- насколько сайт продаёт услуги B2B-клиенту;
- какие элементы можно использовать как ориентир для своего сайта.

Важно:
- Верни только JSON без пояснений и markdown
- Каждый массив должен содержать 4-6 конкретных пунктов
- Пиши на русском языке
- Будь конкретен и практичен
- Давай actionable рекомендации
- Не придумывай факты, которых нет на скриншоте или в переданном тексте
- Если информации недостаточно, прямо укажи: "На сайте не обнаружено..."
- Не утверждай, что компания занимается рефрижераторными перевозками, если это явно не видно
- Не называй отсутствие рефрижераторных перевозок слабой стороной, если компания не позиционируется как рефрижераторный перевозчик
- Отличай факт от предположения"""

        start_time = time.time()
        logger.info("  Отправка скриншота в Vision API...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Проведи комплексный конкурентный анализ этого сайта:\n\n{context}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{screenshot_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.7,
                max_tokens=3000
            )
            
            elapsed = time.time() - start_time
            logger.info(f"  ✓ Ответ получен за {elapsed:.2f} сек")
            
            content = response.choices[0].message.content
            logger.info(f"  Длина ответа: {len(content)} символов")
            
            data = self._parse_json_response(content)
            
            result = CompetitorAnalysis(
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                unique_offers=data.get("unique_offers", []),
                recommendations=data.get("recommendations", []),
                summary=data.get("summary", "")
            )
            
            logger.info(f"  Результат:")
            logger.info(f"    - Сильных сторон: {len(result.strengths)}")
            logger.info(f"    - Слабых сторон: {len(result.weaknesses)}")
            logger.info(f"    - УТП: {len(result.unique_offers)}")
            logger.info(f"    - Рекомендаций: {len(result.recommendations)}")
            logger.info(f"  Резюме: {result.summary[:100]}...")
            logger.info("=" * 50)
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"  ✗ Ошибка Vision API за {elapsed:.2f} сек: {e}")
            logger.error("=" * 50)
            raise


# Глобальный экземпляр
logger.info("Создание глобального экземпляра OpenAI сервиса...")
openai_service = OpenAIService()